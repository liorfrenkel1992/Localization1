# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:46:23 2020

@author: bejellh
"""

import os
import sys
import csv
import argparse

import soundfile as sf
import scipy.io as sio
from scipy.signal import medfilt
from pyroomacoustics.doa.music import MUSIC
from pyroomacoustics.doa.srp import SRP

import matplotlib.pyplot as plt
import numpy as np
import hdf5storage

import keras
from keras.models import model_from_json, load_model
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from unet import UNet

def parseArgs():
    parser = argparse.ArgumentParser(description='Localization')
    parser.add_argument('--mode', type=str, default='test',
                        help='Mode, i.e., train, val or test (default: test)')
    parser.add_argument('--s_path', type=str, default='data/raw',
                        help='Speakers .wav files directory (default: data/raw)')
    parser.add_argument('--csv_path', type=str, default='data/raw',
                        help='csv directory (default: data/raw)')
    parser.add_argument('--soumitro_path', type=str, default='/mnt/dsi_vol1/users/frenkel2/data/localization/hodaya/Single-speaker-localization',
                        help='soumitro model directory (default: /mnt/dsi_vol1/users/frenkel2/data/localization/hodaya/Single-speaker-localization)')
    parser.add_argument('--checkpoints_path', type=str, default='src/visualization/outputs', dest='ckpt_path',
                        help='checkpoints directory (default: src/visualization/outputs)')
    parser.add_argument('-plot', action="store_true", dest='plot',
                        help='plot figures')
    parser.set_defaults(plot=True)
    parser.add_argument('--bs', type=int, dest='batch_size', default=32,
                        help='batch size of the imported model (default: 32)')
    parser.add_argument('--lr', type=str, dest='lr', default='1e-05',
                        help='learning rate of the imported model (default: 1e-05)')
    parser.add_argument('--optim', type=str, dest='optim', default='adam',
                        help='optimization method of the imported model (default: adam)')
    parser.add_argument('--k', type=int, dest='K', default='512',
                        help='STFT size (default: 512)')
    parser.add_argument('--frame_size', type=int, dest='frame_size', default='256',
                        help='frame size (default: 256)')
    parser.add_argument('--n_classes', type=int, dest='n_classes', default='37',
                        help='number of possible predicted DOA (default: 37)')
    parser.add_argument('--noise_phase', type=int, dest='noise_phase', default='270',
                        help='chosen phase for noise (default: 270)')
    parser.add_argument('--ref_channel', type=int, dest='ref_channel', default='2',
                        help='reference channel of mics array (default: 2)')
    parser.add_argument('--min_amp', type=int, dest='MIN_AMP', default='10000',
                        help='minimum amplitude threshold for STFT (default: 10000)')
    parser.add_argument('--amp_fac', type=int, dest='AMP_FAC', default='10000',
                        help='amplitude factor for STFT (default: 10000)')
    parser.add_argument('--threshold', type=int, dest='THRESHOLD', default='40',
                        help='threshold for max amplitude for VAD (default: 40)')
    parser.add_argument('--placements_apart', type=int, dest='placements_apart', default='4',
                        help='minimum number of angles between speakers (default: 4)')
    return parser.parse_args()


def stft(x, nfft=512, overlap=0.75, flag='true'):
    x = x.reshape(x.size,1) # make a column vector for ease later
    nx = x.size
    dM = int(nfft*(1-overlap))
    dN = 1

    window=np.hanning
    win = window(nfft)
    win = win.reshape(win.size,1)
    #find analysis window for the above synthesis window
    #figure out number of columns for offsetting the signal
    #this may truncate the last portion of the signal since we'd
    #rather not append zeros unnecessarily - also makes the fancy
    #indexing that follows more difficult.
    ncol = int(np.fix((nx-nfft)/dM+1))
    y = np.zeros((nfft,ncol))
    #now stuff x into columns of y with the proper offset
    #should be able to do this with fancy indexing!
    colindex = np.arange(0,ncol)*dM
    colindex = colindex.reshape(1,ncol)
    rowindex = (np.arange(0,nfft)).reshape(nfft,1)
    rowindex_rep = (np.tile(rowindex,(1,ncol))).astype(int)
    colindex_rep = (np.tile(colindex,(nfft,1))).astype(int)
    indx = rowindex_rep + colindex_rep
    
    for ii in range(0,nfft):
        idx = (indx[ii,:]).reshape(indx.shape[1])
        y[ii,:] = (x[idx]).reshape(len(idx))

    #Apply the window to the array of offset signal segments.
    win_rep = np.tile(win,(1,ncol))
    y = np.multiply(win_rep,y)
    #now fft y which does the columns
    N = int(nfft/dN)
    for k in range(1,dN):
       y[0:N,:] = y[0:N,:] + y[np.arange((0,N))+k*N,:]
    
    y = y[0:N,:]
    y = np.fft.fft(y,axis=0)
    
    if flag=='true':
        if not np.any(x.imag):
             y = y[0:int(N/2)+1,:]
    Y=y.T
    return Y


def find_speaker(output, n_class):
    
    ar=np.zeros((n_class,))
    
    for i in range(n_class):
        ar[i]=sum(sum((output==i)*1))
    
    return ar.argsort()[-3:-1]


def main(args):
    # Import data from .csv file
    csv_path = [os.path.join(args.csv_path, csv_file) for csv_file in os.listdir(args.csv_path) if os.path.splitext(os.path.join(args.csv_path, csv_file))[1] == '.csv'][0]
    with open(csv_path, 'r') as csv_file:
        csv_dict = csv.DictReader(csv_file)
        s_path = args.s_path
        wav_list = os.listdir(s_path)
        wav_list = [wav_file for wav_file in wav_list if os.path.splitext(os.path.join(s_path, wav_file))[1] == '.wav']
        all_mic_placements = []
        scenarios_angles = []
        for scenario in csv_dict:
            mic_placements = []
            n_speakers = int(scenario['num_speakers'])
            angles = np.zeros((n_speakers,))
            for s in range(1, n_speakers + 1):
                s_angle = int(scenario['speaker{0}_theta'.format(s)])
                
                if s_angle > 180:
                    s_angle -= 180
                
                angles[s - 1] = int(s_angle / 5)
            n_mics = int(scenario['num_mics'])
            for mic in range(1, n_mics + 1):
                mic_placements.append([float(scenario['mic{0}_x'.format(mic)]), float(scenario['mic{0}_y'.format(mic)]), float(scenario['mic{0}_z'.format(mic)])])
            all_mic_placements.append(np.array(mic_placements))
            scenarios_angles.append(angles)

    eps = sys.float_info.epsilon
    frame_size = args.frame_size
    n_class = args.n_classes
    plot = args.plot
    noise_phase = args.noise_phase
    ref_channel = args.ref_channel
    MIN_AMP = args.MIN_AMP
    AMP_FAC = args.AMP_FAC
    THRESHOLD = args.THRESHOLD
    num_of_placements_apart = args.placements_apart
    specFixedVar_1ch = 3
    
    # Set fft window properties
    K = args.K
    spec_size = int(K/2)
    win_dir = 'src/data/synt_win_512.mat'
    mat_contents = sio.loadmat(win_dir)
    synt_win = mat_contents['synt_win']

    # Import our model
    model_main_dir = args.ckpt_path
    model_name = 'unet_realRTF_imgRTF_spect_16Khz_4micsArray_bs_{0}_lr_{1}_{2}_Lior.ckpt'.format(args.batch_size, args.lr, args.optim)
    #model_name = 'unet_realRTF_imgRTF_spect_16Khz_4micsArray_bs_{0}_lr_{1}_Lior.ckpt'.format(args.batch_size, args.lr)
    model_5deg_4mic_realRTF_imgRTF = UNet.load_from_checkpoint(os.path.join(model_main_dir, model_name))

    # Import and compile soumitro model
    soumitro_path = args.soumitro_path
    json_file = open(os.path.join(soumitro_path, 'Model_mul_CNN_paper.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_soumitro = model_from_json(loaded_model_json)
    model_soumitro.load_weights(os.path.join(soumitro_path, 'Weights_mul_CNN_paper.h5'))
    lrate = 0.001
    adam = Adam(lr =lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_soumitro.compile(loss='categorical_crossentropy', optimizer= adam)
    
    # Initialize models eval data
    models = [model_5deg_4mic_realRTF_imgRTF, model_soumitro]
    model_names = ['model_5deg_4mic_realRTF_imgRTF', 'model_soumitro', 'SRP', 'MUSIC']

    accuracy_by_batch = {model_names[0]:np.zeros((len(wav_list))),model_names[1]:np.zeros((len(wav_list))),model_names[2]:np.zeros((len(wav_list)))}
    accuracy_by_frame_1spk = {model_names[0]:np.zeros((len(wav_list))),model_names[1]:np.zeros((len(wav_list))),model_names[2]:np.zeros((len(wav_list)))}
    accuracy_by_frame_2spk = {model_names[0]:np.zeros((len(wav_list))),model_names[1]:np.zeros((len(wav_list))),model_names[2]:np.zeros((len(wav_list)))}
    accuracy_by_frame_1spk_medfilt = {model_names[0]:np.zeros((len(wav_list))),model_names[1]:np.zeros((len(wav_list))),model_names[2]:np.zeros((len(wav_list)))}
    accuracy_by_frame_2spk_medfilt = {model_names[0]:np.zeros((len(wav_list))),model_names[1]:np.zeros((len(wav_list))),model_names[2]:np.zeros((len(wav_list)))}

    count_spk = np.zeros((3, len(wav_list)))

    for wav_inx, wav_file in enumerate(wav_list):

        wav_file = wav_list[45]  # Scenario with 2 speakers
        full_name = os.path.join(s_path, wav_file)
        print(full_name)  
        
        wav_scenario = int(wav_file.split('_')[1])
        wav_name = os.path.splitext(wav_file)[0]
                                                    
        num_speakers = int(wav_file.split('_')[2].split('.')[0])
        
        for ii in range(len(models)):
            doa_final = []
            o_final = []
            o_reshape_with_vad_final = []
            
            # Our model
            model = models[ii]
            model_name = model_names[ii]
            s,fs = sf.read(full_name)
            # s = s/np.std(s)
            n_mics = s.shape[-1]

            if model_name == 'model_5deg_4mic_realRTF_imgRTF':
                # true_label = hdf5storage.loadmat(os.path.join(args.s_path, wav_name + '_trueLABEL.mat'))
                # true_label = true_label['label_mat']
                true_label = hdf5storage.loadmat(os.path.join(args.s_path, wav_name + '.mat'))
                true_label = true_label['y_train']
                true_label = np.where(true_label > 180, 360 - true_label, true_label)  # Converting to range [0,180]
                true_label[true_label == -1] = noise_phase  # Noise label was defined as -1
                true_label = (true_label/5).astype(int)  # Converting labels to range [0,37]
                true_label = true_label[0:spec_size,:]
                phases = np.unique(true_label)
                phases_deg = (phases[:-1]-1)*5  # Without noise label
                true_label_reshape_5deg = np.zeros((true_label.shape[0], true_label.shape[1], 37))
                
                for phase in range(phases.shape[0] - 1):
                    true_label_reshape_5deg[:,:,int(phases[phase])][true_label[:, :, ref_channel]== phases[phase]] = 1
                true_doa_5deg = np.sum(true_label_reshape_5deg, axis=0)                
                if plot:
                    plt.figure()
                    plt.imshow(true_doa_5deg.T,aspect='auto')
                    plt.savefig(os.path.join(args.s_path, 'True doa 5deg_'+str(wav_scenario) + '_model_'+model_name))
                    plt.close()

                for flip in ['False', 'True']:
                    s = s[:,::-1]  # Flip mics
                    temp = stft(s[:, 1], nfft=K, overlap=0.75).T  # STFT to 2nd mic
                
                    speech_mix_spec = np.abs(temp)
                    speech_mix_spec = np.maximum(speech_mix_spec, np.max(speech_mix_spec)/MIN_AMP)
                    speech_mix_spec = 20.*np.log10(speech_mix_spec*AMP_FAC)  # Convert spec to log scale
                
                    max_mag = np.max(speech_mix_spec)
                    speech_VAD = (speech_mix_spec > (max_mag - THRESHOLD))  # Create VAD by threshold      

                    long = int(np.ceil(temp.shape[1]/frame_size))
                    temp = np.concatenate((temp,np.zeros((spec_size+1, long*frame_size-temp.shape[1]))), axis=1)  # Complete spec to complete double of frame_size

                    # Divide spec to equal parts of size frame_size
                    x_test = np.zeros((long,spec_size+1, frame_size, s.shape[1]), dtype=complex)
                    for i in range(s.shape[1]):
                        temp_stft = stft(s[:,i], nfft=K, overlap=0.75).T
                        for j in range(long - 1):
                            x_test[j,:,:,i] = temp_stft[:,(j*frame_size):(j*frame_size)+frame_size]
                        x_test[long - 1, :, 0:temp_stft.shape[1] - ((j * frame_size) + frame_size), i] = temp_stft[:, ((long-1)*frame_size):temp_stft.shape[1]]
                    
                    x_spec = np.log(eps + np.abs(x_test[:, 0:spec_size, :, ref_channel]))  # Convert spec to log values

                    ### Make log values in range [-1,1]
                    # x_spec_min = np.expand_dims(np.expand_dims(x_spec.min(axis=(1,2)),1),-1)
                    # x_spec_min = np.tile(x_spec_min, (1,spec_size,frame_size))
                    # x_spec_max = np.expand_dims(np.expand_dims(x_spec.max(axis=(1,2)),1),-1)
                    # x_spec_max = np.tile(x_spec_max, (1,spec_size,frame_size))
                    # x_spec = ((x_spec-x_spec_min)/(x_spec_max-x_spec_min))*2-1
                    # del x_spec_max, x_spec_min
                    # x_spec = np.expand_dims(x_spec,3)
                    
                    
                    ## varaince 1 for every spectrogram
                    x_spec_std = np.expand_dims(np.expand_dims(x_spec.std(axis=(1,2)),1),-2)
                    x_spec_std = np.tile(x_spec_std, (1,spec_size,frame_size))
                    # x_spec_std = x_spec.std(axis=(1, 2))  # Divide all by 1 mic std
                    x_spec = x_spec*((np.sqrt(specFixedVar_1ch)/(eps+x_spec_std)))
                    del x_spec_std
                    
                    ### Make log values in range [0,1]
                    x_spec = (x_spec - x_spec.min()) / (x_spec.max() - x_spec.min())
                    x_spec = np.expand_dims(x_spec, 3)
                    
                    # RTF with 3rd mic
                    x_rtf = x_test / (np.expand_dims(x_test[:,:,:,ref_channel], 3) + eps)
                    x_rtf = np.concatenate((x_rtf[:,:,:,:ref_channel], x_rtf[:,:,:,ref_channel+1:]), axis=-1)
                    x_rtf = x_rtf[:, :spec_size, :, :]
                    
                    # x_rtf_std = np.expand_dims(np.expand_dims(x_rtf.std(axis=(1,2)),1),-2)
                    # x_rtf_std = np.tile(x_rtf_std, (1, spec_size, frame_size, 1))
                    x_rtf_std = x_rtf.std(axis=(0, 1))[ref_channel - 1]  # Divide all by 1st mic std
                    x_rtf = x_rtf * ((np.sqrt(1) / (eps + x_rtf_std)))  # Normalizing RTF with its std
                    
                    x_real = np.real(x_rtf)
                    x_image = np.imag(x_rtf)
                    
                    ### Make rtf values in range [0,1]
                    x_real = (x_real - x_real.min()) / (x_real.max() - x_real.min())
                    x_image = (x_image - x_image.min()) / (x_image.max() - x_image.min())
        
                    x_test_final=np.concatenate((x_real,x_image,x_spec),axis=-1)

                    # Predict with our model
                    startt = 0
                    endd = startt + long
                    x_test_final = torch.from_numpy(x_test_final.transpose((0, 3, 1, 2))).type(torch.FloatTensor)
                    o = model(x_test_final[startt:endd, 0:spec_size, :, :])#.reshape((1,x.shape[1],96,8)))
                    # o = o[:, :-1, :, :]  # Ignore noise labels
                    o = F.softmax(o, dim=1)
                    o = o.detach().numpy().transpose((0, 2, 3, 1))
                    
                    if (flip=='False'):
                        o = o
                    else:
                        o = o[:,:,:,::-1]

                    stop_frame = temp_stft.shape[1]
                    speech_VAD = speech_VAD[:256, :stop_frame]
                    o_reshape = np.reshape(o.transpose(1, 0, 2, 3),(spec_size, frame_size*long, n_class), 'C')  # Bring back to original spec form
                    o_reshape_2 = o_reshape[:, :stop_frame, :]
                    o_reshape_with_vad = o_reshape_2 * np.expand_dims(speech_VAD, 2)
                    o_argmax = np.argmax(o_reshape_with_vad[::-1], axis=-1)
                    o_unique = np.unique(o_argmax)
                    o_doa_count = np.zeros((o_reshape_with_vad.shape[1], o_reshape_with_vad.shape[2]))
                    for angle in o_unique:
                        o_doa_count[:, angle] = np.count_nonzero(o_argmax == angle, 0)
                    
                    if plot:
                        plt.figure()
                        plt.imshow(o_doa_count.T, aspect='auto')
                        plt.savefig(os.path.join('reports/figures', 'argmax_count_' + str(wav_scenario) + '_model_' + model_name))
                        plt.close()
                    
                    
                    
                    doa = np.sum(o_reshape_with_vad, axis=0)
                    #doa = np.sum(o_reshape_2, axis=0)

                    doa_final.append(doa)
                    o_final.append(o)
                    o_reshape_with_vad_final.append(o_reshape_with_vad)
                    #o_reshape_with_vad_final.append(o_reshape_2)

                
                # Concat back the original and flipped predicion
                o = np.concatenate((o_final[0][:, :, :, :19], o_final[1][:, :, :, -18:]), axis=-1)
                o_reshape_with_vad_final = np.concatenate((o_reshape_with_vad_final[0][:, :, :19], o_reshape_with_vad_final[1][:, :, -18:]), axis=-1)
                doa_final = np.concatenate((doa_final[0][:,:19],doa_final[1][:,-18:]),axis=-1)
                
                """
                doa_final = doa_final[0]
                o_final = o_final[0]
                o_reshape_with_vad_final = o_reshape_with_vad_final[0]
                """
                n_speakers
                            
                if plot:
                    plt.figure()
                    plt.imshow(doa_final.T, aspect='auto')
                    plt.savefig(os.path.join(args.s_path, 'doaFinal_' + str(wav_scenario) + '_model_' + model_name))
                    plt.close()
                
                ooo = np.argmax(o, axis=3)  # Final prediction from model by hard decision
                    
                final_in = np.zeros((spec_size + 1, 0))
                final_out = np.zeros((spec_size , 0))
                final_target = np.zeros((spec_size, 0))
                soft_out_1 = np.zeros((spec_size, 0))
                temp = np.zeros((spec_size, 0))
                soft_out = np.zeros((o.shape[-1], spec_size, frame_size * len(ooo)))

                for i in range(len(ooo)):
                    final_out = np.concatenate((final_out, ooo[i, :, :]), axis=1)
                    temp_spec = x_test[startt + i, :, :, ref_channel]
                    final_in = np.concatenate((final_in, temp_spec), axis=1)

                spkrs = find_speaker(final_out, n_class)
                for theta in range(o.shape[-1]):
                    temp = np.zeros((spec_size, 0))
                    for i in range(len(ooo)):
                        a = o[i, :, :, theta]
                        temp = np.concatenate((temp, a), axis=1)
                    soft_out[theta, :, :] = temp
                            
                final_soft = np.argmax(soft_out, axis=0)
                
                low_final_soft = final_soft[0:60, :]
                spks = np.zeros((o.shape[-1],))
                
                final_soft = (final_soft[:, 0:stop_frame] + 1) * speech_VAD[:256, 0:stop_frame]
                
                for i in range(o.shape[-1]):
                    spks[i] = soft_out[i, :, 0:stop_frame].sum()

                #two_peaks=spks.argsort()[-2:]
                two_peaks = spks.argsort()[-num_speakers:]
                # Predicted speakers are at least 10 deg from each other
                idx = 1
                if len(two_peaks) == 2:
                    while (abs(two_peaks[0] - two_peaks[1]) < num_of_placements_apart - 1):
                        two_peaks[1] = spks.argsort()[-2 - idx]
                        idx = idx + 1
                
                

            # Soumitro model 
            elif model_name == 'model_soumitro':
                true_label = hdf5storage.loadmat(os.path.join(args.s_path, wav_name + '_trueLABELsoumitro.mat'))
                true_label = true_label['label_mat_soumitro']
                true_label = np.where(true_label > 180, 360 - true_label , true_label)
                true_label[true_label == -1] = noise_phase
                true_label = (true_label/5).astype(int)
                true_label = true_label[0:spec_size, :]
                phases = np.unique(true_label)
                phases_deg = (phases[1:]-1)*5
                true_label_reshape_5deg = np.zeros((true_label.shape[0], true_label.shape[1], 37))

                for phase in range(phases.shape[0] - 1):
                    true_label_reshape_5deg[:, :, int(phases[phase])][true_label == phases[phase]] = 1 
                true_doa_5deg = np.sum(true_label_reshape_5deg, axis=0)                
                if plot:
                    plt.figure()
                    plt.imshow(true_doa_5deg.T, aspect='auto')
                    plt.savefig(os.path.join(args.s_path, 'True doa 5deg_' + str(wav_scenario) + '_model_' + model_name))
                    plt.close()

                temp = stft(s[:,1], nfft=K, overlap=0.5).T
                S = np.zeros((n_mics, spec_size + 1, temp.shape[1]), dtype='complex')
                for mic_idx in range(n_mics):
                    S[mic_idx, :, :] = stft(s[:, mic_idx], nfft=K, overlap=0.5).T
                    
                angles = np.angle(S)
                angles = np.transpose(angles, (2, 1, 0))
                x_test = np.expand_dims(angles, 1)
                Output = model.predict(x_test)              
                doa_final = Output[:, ::-1]
                if plot:
                    plt.figure()
                    plt.imshow(doa_final.T, aspect='auto')
                    plt.savefig(os.path.join(args.s_path, 'doaFinal_'+str(wav_scenario)+ '_model_' + model_name))
                    plt.close()
                spks = np.zeros((doa_final.shape[-1],))
                                    
                for i in range(doa_final.shape[-1]):
                    spks[i] = doa_final[:, i].sum()

                #two_peaks=spks.argsort()[-2:]
                two_peaks = spks.argsort()[-num_speakers:]
                idx = 1
                while (abs(two_peaks[0] - two_peaks[1]) < num_of_placements_apart - 1):
                    two_peaks = [spks.argsort()[-1], spks.argsort()[-2 - idx]]
                    idx = idx + 1
            
            doa_final_norm = (doa_final - doa_final.min()) / (doa_final.max() - doa_final.min())

            if len(two_peaks) == 2:
                for idx_doa in range(doa_final_norm.shape[0]):
                    doa_final_argsort = np.argsort(doa_final_norm[idx_doa,:])
                    for idx in range(1,6):
                        if ((abs(doa_final_argsort[-idx]-two_peaks[0])<num_of_placements_apart-1 or abs(doa_final_argsort[-idx]-two_peaks[1])<num_of_placements_apart-1) and (doa_final_norm[idx_doa,doa_final_argsort[-idx]]>0.1)):
                            minidx = np.argmin([abs(doa_final_argsort[-idx]-two_peaks[0]),abs(doa_final_argsort[-idx]-two_peaks[1])])
                            minidx = two_peaks[minidx]
                            if (minidx != doa_final_argsort[-idx]):
                                doa_final_norm[idx_doa,minidx] = doa_final_norm[idx_doa,minidx]+doa_final_norm[idx_doa,doa_final_argsort[-idx]]
                                doa_final_norm[idx_doa,doa_final_argsort[-idx]] = 0
                        else:
                            doa_final_norm[idx_doa,doa_final_argsort[-idx]] = 0

            doa_final_norm[doa_final_norm<0.03] = 0
            doa_final_norm = doa_final_norm/((np.tile(np.sum(doa_final_norm,axis=1),(doa_final_norm.shape[1],1))+eps).T)


            ## smooth every phase using med filt
            doa_final_smooth = np.zeros_like(doa_final_norm)
            for phase_idx in range(doa_final_norm.shape[1]):
                doa_final_smooth[:,phase_idx] = medfilt(doa_final_norm[:,phase_idx],kernel_size=5)
            if plot:
                plt.figure()
                plt.imshow(doa_final_norm.T,aspect='auto')
                plt.savefig(os.path.join(args.s_path, 'doaFinalNorm_'+str(wav_scenario)+'_model_'+model_name))
                plt.close()
                plt.figure()
                plt.imshow(doa_final_smooth.T,aspect='auto')
                plt.savefig(os.path.join(args.s_path, 'doaFinalSmooth_'+str(wav_scenario)+ '_model_'+model_name))
                plt.close()                
            
            final_soft_doa = doa_final_norm


            frame_estimated_phases = np.zeros((final_soft_doa.shape[0],2))
            frame_true_phase = np.zeros((true_doa_5deg.shape[0],2))
            
            for idx in range(final_soft_doa.shape[0]):
                frame_estimated_phases_argsort = np.argsort(final_soft_doa[idx,:],axis=-1)
                true_doa_5deg_argsort = np.argsort(true_doa_5deg[idx,:],axis=-1)
                if (np.sum(final_soft_doa[idx,:],axis=-1)==0 or np.sum(true_doa_5deg[idx,:],axis=-1)==0):  #frames of VAD
                    frame_estimated_phases[idx,:] = [noise_phase, noise_phase]
                    frame_true_phase[idx,:] = [noise_phase, noise_phase]
                elif (true_doa_5deg[idx,:][true_doa_5deg_argsort[-1]]>0 and true_doa_5deg[idx,:][true_doa_5deg_argsort[-2]]>0 and final_soft_doa[idx,:][frame_estimated_phases_argsort[-1]]>0 and final_soft_doa[idx,:][frame_estimated_phases_argsort[-2]]>0):    ## There are two speakers
                    two_peaks_est = frame_estimated_phases_argsort[-2:].astype(int)
                    kk = 1
                    while (abs(two_peaks_est[0]-two_peaks_est[1])<num_of_placements_apart-1):
                        two_peaks_est=np.array([int(frame_estimated_phases_argsort[-1]),int(frame_estimated_phases_argsort[-2-kk])])
                        kk=kk+1
                    two_peaks_est.sort()
                    frame_estimated_phases[idx,:] = 5*two_peaks_est
                    two_peaks_true = np.array([true_doa_5deg_argsort[-1],true_doa_5deg_argsort[-2]])
                    two_peaks_true.sort()
                    frame_true_phase[idx,:] = 5*two_peaks_true
                else:   #only one speaker
                    if (final_soft_doa[idx,:][frame_estimated_phases_argsort[-1]]==final_soft_doa[idx,:][frame_estimated_phases_argsort[-2]]):        ## When there is no maximum but two maximums we will take the one according to the previous frame
                        frame_estimated_phases[idx,:] = [frame_estimated_phases[idx-1,0],noise_phase]
                    else:
                        frame_estimated_phases[idx,:] = [5*frame_estimated_phases_argsort[-1],noise_phase]
                    frame_true_phase[idx,:] = [5*true_doa_5deg_argsort[-1],noise_phase]                
                    
            
            vad_by_frame = np.ones_like(frame_true_phase)
            vad_by_frame[frame_true_phase==noise_phase] = 0
            tot_num_vad_by_frame = vad_by_frame.sum(axis=1)
            
            if ii == 0:
                r = np.array(range(len(tot_num_vad_by_frame)))
                idxs_2spk = r[tot_num_vad_by_frame==2]
                idxs_1spk = r[tot_num_vad_by_frame==1]
                doa_srp = SRP(all_mic_placements[wav_scenario - 1].T, fs, K, azimuth=np.linspace(0.,180.,37)*np.pi/180)
                doa_music = MUSIC(all_mic_placements[wav_scenario - 1].T, fs, K, azimuth=np.linspace(0.,180.,37)*np.pi/180)
                
                X=np.zeros((spec_size+1,0,4))

                for i in range(len(ooo)):
                    temp_spec=x_test[startt+i,:,:,:]
                    X=np.concatenate((X,temp_spec),axis=1) 
                
                X = X[:,0:stop_frame,:]
            
            tot_num_vad_by_frame_1spk = int(np.sum(tot_num_vad_by_frame[tot_num_vad_by_frame==1]))
            if tot_num_vad_by_frame_1spk == 0:
                count_spk[1,wav_inx] = 0
                accuracy_by_frame_1speaker_mean_soumitro = 0
            else:
                accuracy_by_frame_1speaker_tmp = np.abs(frame_true_phase[tot_num_vad_by_frame==1]-frame_estimated_phases[tot_num_vad_by_frame==1])
                accuracy_by_frame_1speaker_mean_soumitro = np.sum(accuracy_by_frame_1speaker_tmp)/tot_num_vad_by_frame_1spk
                count_spk[1,wav_inx] = tot_num_vad_by_frame_1spk
                if ii == 0:
                    X_1spk = X[:,idxs_1spk]
                    srp_1spk_est = np.zeros(len(idxs_1spk))
                    music_1spk_est = np.zeros(len(idxs_1spk))
                    for idx1spk in range(len(idxs_1spk)):
                        doa_srp.locate_sources(np.expand_dims(X_1spk[:,idx1spk,:].T,-1),num_src=1)
                        srp_1spk_est[idx1spk] = doa_srp.azimuth_recon*180/np.pi
                        doa_music.locate_sources(np.expand_dims(X_1spk[:,idx1spk,:].T,-1),num_src=1)
                        music_1spk_est[idx1spk] = doa_music.azimuth_recon*180/np.pi
                    accuracy_by_frame_1speaker_srp = np.abs(frame_true_phase[tot_num_vad_by_frame==1][:,0]-srp_1spk_est)
                    accuracy_by_frame_1speaker_srp_mean = np.sum(accuracy_by_frame_1speaker_srp)/tot_num_vad_by_frame_1spk
                    accuracy_by_frame_1speaker_music = np.abs(frame_true_phase[tot_num_vad_by_frame==1][:,0]-music_1spk_est)
                    accuracy_by_frame_1speaker_music_mean = np.sum(accuracy_by_frame_1speaker_music)/tot_num_vad_by_frame_1spk

            tot_num_vad_by_frame_2spk = int(np.sum(tot_num_vad_by_frame[tot_num_vad_by_frame==2]))
            if tot_num_vad_by_frame_2spk==0:
                count_spk[2,wav_inx] = 0
                accuracy_by_frame_2speaker_srp_mean = 0
            else:
                accuracy_by_frame_2speaker_tmp = np.abs(frame_true_phase[tot_num_vad_by_frame==2]-frame_estimated_phases[tot_num_vad_by_frame==2])
                accuracy_by_frame_2speaker_mean = np.sum(accuracy_by_frame_2speaker_tmp)/tot_num_vad_by_frame_2spk

                count_spk[2,wav_inx] = int(tot_num_vad_by_frame_2spk/2)
                
                if ii == 0:
                    X_2spk = X[:,idxs_2spk]
                    srp_2spk_est = np.zeros((len(idxs_2spk),2))
                    music_2spk_est = np.zeros((len(idxs_2spk),2))
                    
                    for idx2spk in range(len(idxs_2spk)):
                        doa_srp.locate_sources(np.expand_dims(X_2spk[:,idx2spk,:].T,-1),num_src=2)
                        srp_2spk_est[idx2spk,:] = doa_srp.azimuth_recon*180/np.pi
                        doa_music.locate_sources(np.expand_dims(X_2spk[:,idx2spk,:].T,-1),num_src=2)
                        music_2spk_est[idx2spk,:] = doa_music.azimuth_recon*180/np.pi
                    accuracy_by_frame_2speaker_srp = np.abs(frame_true_phase[tot_num_vad_by_frame==2]-srp_2spk_est)
                    accuracy_by_frame_2speaker_srp_mean = np.sum(accuracy_by_frame_2speaker_srp)/tot_num_vad_by_frame_2spk
                    accuracy_by_frame_2speaker_music = np.abs(frame_true_phase[tot_num_vad_by_frame==2]-music_2spk_est)
                    accuracy_by_frame_2speaker_music_mean = np.sum(accuracy_by_frame_2speaker_music)/tot_num_vad_by_frame_2spk                    
            
            accuracy_by_frame_1spk[model_names[ii]][wav_inx] = accuracy_by_frame_1speaker_mean
            accuracy_by_frame_2spk[model_names[ii]][wav_inx] = accuracy_by_frame_2speaker_mean
            
            if ii == 0: 
                accuracy_by_frame_1spk[model_names[2]][wav_inx] = accuracy_by_frame_1speaker_srp_mean
                accuracy_by_frame_1spk[model_names[3]][wav_inx] = accuracy_by_frame_1speaker_music_mean
                accuracy_by_frame_2spk[model_names[2]][wav_inx] = accuracy_by_frame_2speaker_srp_mean
                accuracy_by_frame_2spk[model_names[3]][wav_inx] = accuracy_by_frame_2speaker_music_mean
            
            
            ##accuracy with medfilt
            final_soft_doa = doa_final_smooth

            frame_estimated_phases = np.zeros((final_soft_doa.shape[0],2))
            frame_true_phase = np.zeros((true_doa_5deg.shape[0],2))
            
            for idx in range(final_soft_doa.shape[0]):
                frame_estimated_phases_argsort = np.argsort(final_soft_doa[idx,:],axis=-1)
                true_doa_5deg_argsort = np.argsort(true_doa_5deg[idx,:],axis=-1)
                if (np.sum(final_soft_doa[idx,:],axis=-1)==0 or np.sum(true_doa_5deg[idx,:],axis=-1)==0):  #frames of VAD
                    frame_estimated_phases[idx,:] = [noise_phase, noise_phase]
                    frame_true_phase[idx,:] = [noise_phase, noise_phase]
                elif (true_doa_5deg[idx,:][true_doa_5deg_argsort[-1]]>0 and true_doa_5deg[idx,:][true_doa_5deg_argsort[-2]]>0 and final_soft_doa[idx,:][frame_estimated_phases_argsort[-1]]>0 and final_soft_doa[idx,:][frame_estimated_phases_argsort[-2]]>0):    ## There are two speakers
                    two_peaks_est = frame_estimated_phases_argsort[-2:].astype(int)
                    kk = 1
                    while (abs(two_peaks_est[0]-two_peaks_est[1])<num_of_placements_apart-1):
                        two_peaks_est=np.array([int(frame_estimated_phases_argsort[-1]),int(frame_estimated_phases_argsort[-2-kk])])
                        kk=kk+1
                    two_peaks_est.sort()
                    frame_estimated_phases[idx,:] = 5*two_peaks_est
                    two_peaks_true = np.array([true_doa_5deg_argsort[-1],true_doa_5deg_argsort[-2]])
                    two_peaks_true.sort()
                    frame_true_phase[idx,:] = 5*two_peaks_true
                else:   #only one speaker
                    if (final_soft_doa[idx,:][frame_estimated_phases_argsort[-1]]==final_soft_doa[idx,:][frame_estimated_phases_argsort[-2]]):        ## When there is no maximum but two maximums we will take the one according to the previous frame
                        frame_estimated_phases[idx,:] = [frame_estimated_phases[idx-1,0],noise_phase]
                    else:
                        frame_estimated_phases[idx,:] = [5*frame_estimated_phases_argsort[-1],noise_phase]
                    frame_true_phase[idx,:] = [5*true_doa_5deg_argsort[-1],noise_phase]                
                    
            
            vad_by_frame = np.ones_like(frame_true_phase)
            vad_by_frame[frame_true_phase==noise_phase] = 0
            tot_num_vad_by_frame = vad_by_frame.sum(axis=1)
            
            if ii == 0:
                r = np.array(range(len(tot_num_vad_by_frame)))
                idxs_2spk = r[tot_num_vad_by_frame==2]
                idxs_1spk = r[tot_num_vad_by_frame==1]
                doa_srp = SRP(all_mic_placements[wav_scenario - 1].T, fs, K, azimuth=np.linspace(0.,180.,37)*np.pi/180, num_src=2)
                doa_music = MUSIC(all_mic_placements[wav_scenario - 1].T, fs, K, azimuth=np.linspace(0.,180.,37)*np.pi/180, num_src=2)
                
                X=np.zeros((spec_size+1,0,4))

                for i in range(len(ooo)):
                    temp_spec=x_test[startt+i,:,:,:]
                    X=np.concatenate((X,temp_spec),axis=1) 
                
                X = X[:,0:stop_frame,:]
            
            tot_num_vad_by_frame_1spk = int(np.sum(tot_num_vad_by_frame[tot_num_vad_by_frame==1]))
            if tot_num_vad_by_frame_1spk == 0:
                count_spk[1,wav_inx] = 0
                accuracy_by_frame_1speaker_mean = 0
            else:
                accuracy_by_frame_1speaker_tmp = np.abs(frame_true_phase[tot_num_vad_by_frame==1]-frame_estimated_phases[tot_num_vad_by_frame==1])
                accuracy_by_frame_1speaker_mean = np.sum(accuracy_by_frame_1speaker_tmp)/tot_num_vad_by_frame_1spk
                count_spk[1,wav_inx] = tot_num_vad_by_frame_1spk
                if ii == 0:
                    X_1spk = X[:,idxs_1spk]
                    srp_1spk_est = np.zeros(len(idxs_1spk))
                    music_1spk_est = np.zeros(len(idxs_1spk))
                    for idx1spk in range(len(idxs_1spk)):
                        doa_srp.locate_sources(np.expand_dims(X_1spk[:,idx1spk,:].T,-1),num_src=1)
                        srp_1spk_est[idx1spk] = doa_srp.azimuth_recon*180/np.pi
                        doa_music.locate_sources(np.expand_dims(X_1spk[:,idx1spk,:].T,-1),num_src=1)
                        music_1spk_est[idx1spk] = doa_music.azimuth_recon*180/np.pi
                    accuracy_by_frame_1speaker_srp = np.abs(frame_true_phase[tot_num_vad_by_frame==1][:,0]-srp_1spk_est)
                    accuracy_by_frame_1speaker_srp_mean = np.sum(accuracy_by_frame_1speaker_srp)/tot_num_vad_by_frame_1spk
                    accuracy_by_frame_1speaker_music = np.abs(frame_true_phase[tot_num_vad_by_frame==1][:,0]-music_1spk_est)
                    accuracy_by_frame_1speaker_music_mean = np.sum(accuracy_by_frame_1speaker_music)/tot_num_vad_by_frame_1spk
            tot_num_vad_by_frame_2spk = int(np.sum(tot_num_vad_by_frame[tot_num_vad_by_frame==2]))
            if tot_num_vad_by_frame_2spk==0:
                count_spk[2,wav_inx] = 0
                accuracy_by_frame_2speaker_mean = 0
            else:
                accuracy_by_frame_2speaker_tmp = np.abs(frame_true_phase[tot_num_vad_by_frame==2]-frame_estimated_phases[tot_num_vad_by_frame==2])
                accuracy_by_frame_2speaker_mean = np.sum(accuracy_by_frame_2speaker_tmp)/tot_num_vad_by_frame_2spk
                count_spk[2,wav_inx] = int(tot_num_vad_by_frame_2spk/2)
                
                if ii == 0:
                    X_2spk = X[:,idxs_2spk]
                    srp_2spk_est = np.zeros((len(idxs_2spk),2))
                    music_2spk_est = np.zeros((len(idxs_2spk),2))
                    for idx2spk in range(len(idxs_2spk)):
                        doa_srp.locate_sources(np.expand_dims(X_2spk[:,idx2spk,:].T,-1),num_src=2)
                        srp_2spk_est[idx2spk,:] = doa_srp.azimuth_recon*180/np.pi
                        doa_music.locate_sources(np.expand_dims(X_2spk[:,idx2spk,:].T,-1),num_src=2)
                        music_2spk_est[idx2spk,:] = doa_music.azimuth_recon*180/np.pi
                    accuracy_by_frame_2speaker_srp = np.abs(frame_true_phase[tot_num_vad_by_frame==2]-srp_2spk_est)
                    accuracy_by_frame_2speaker_srp_mean = np.sum(accuracy_by_frame_2speaker_srp)/tot_num_vad_by_frame_2spk
                    accuracy_by_frame_2speaker_music = np.abs(frame_true_phase[tot_num_vad_by_frame==2]-music_2spk_est)
                    accuracy_by_frame_2speaker_music_mean = np.sum(accuracy_by_frame_2speaker_music)/tot_num_vad_by_frame_2spk                    
            
            accuracy_by_frame_1spk_medfilt[model_names[ii]][wav_inx] = accuracy_by_frame_1speaker_mean
            accuracy_by_frame_2spk_medfilt[model_names[ii]][wav_inx] = accuracy_by_frame_2speaker_mean
            
            if ii == 0:
                accuracy_by_frame_1spk_medfilt[model_names[2]][wav_inx] = accuracy_by_frame_1speaker_srp_mean
                accuracy_by_frame_1spk_medfilt[model_names[3]][wav_inx] = accuracy_by_frame_1speaker_music_mean
                accuracy_by_frame_2spk_medfilt[model_names[2]][wav_inx] = accuracy_by_frame_2speaker_srp_mean
                accuracy_by_frame_2spk_medfilt[model_names[3]][wav_inx] = accuracy_by_frame_2speaker_music_mean
                


    accuracy_by_frame_1spk_by_room = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    accuracy_by_frame_1spk_mean = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    accuracy_by_frame_2spk_by_room = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    accuracy_by_frame_2spk_mean = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    accuracdy_room2 = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    accuracy_room1_df = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    accuracy_room2_df = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    accuracy_by_frame_by_room = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    accuracy_by_frame_mean = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    count_spk_room1 = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    count_spk_room2 = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    count_spk_room1_sum = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}
    count_spk_room2_sum = {model_names[0]:[],model_names[1]:[], model_names[2]:[],model_names[3]:[]}


    for ii in range(len(model_names)):  
        
        accuracy_by_frame_1spk_by_room[model_names[ii]].append(np.mean(accuracy_by_frame_1spk[model_names[ii]]))
        accuracy_by_frame_1spk_mean[model_names[ii]].append(np.mean(accuracy_by_frame_1spk_by_room[model_names[ii]]))

        accuracy_by_frame_2spk_by_room[model_names[ii]].append(np.mean(accuracy_by_frame_2spk[model_names[ii]]))
        accuracy_by_frame_2spk_mean[model_names[ii]].append(np.mean(accuracy_by_frame_2spk_by_room[model_names[ii]]))
            
        accuracy_by_frame_by_room[model_names[ii]].append(np.mean(accuracy_by_frame[model_names[ii]]))
        accuracy_by_frame_mean[model_names[ii]].append(np.mean(accuracy_by_frame_by_room[model_names[ii]]))

        count_spk_room1[model_names[ii]].append(count_spk)
        
        count_spk_room1_sum[model_names[ii]].append(np.sum(count_spk))
        count_spk_room2[model_names[ii]].append(count_spk)
        count_spk_room2_sum[model_names[ii]].append(np.sum(count_spk))
        
    with open(os.path.join(args.s_path, 'numeric_results2'), 'w') as f:
        f.write('accuracy_by_frame_1spk_mean: ' + str(accuracy_by_frame_1spk_by_room) + '\n')
        f.write('accuracy_by_frame_2spk_mean: ' + str(accuracy_by_frame_2spk_by_room) + '\n')
        f.write('count_spk_room1_sum: ' + str(count_spk_room1_sum) + '\n')
        f.write('count_spk_room2: ' + str(count_spk_room2) + '\n')
        f.write('count_spk_room2_sum: ' + str(count_spk_room2_sum))
        f.close()
        """ 
        accuracy_by_batch_by_room[model_names[ii]].append(np.mean(accuracy_by_batch[model_names[ii]],axis=(1,2)))
        accuracy_by_batch_mean[model_names[ii]].append(np.mean(accuracy_by_batch_by_room[model_names[ii]]))
        
        accuracy_by_frame_1spk_by_room[model_names[ii]].append(np.mean(accuracy_by_frame_1spk[model_names[ii]],axis=(1,2)))
        accuracy_by_frame_1spk_mean[model_names[ii]].append(np.mean(accuracy_by_frame_1spk_by_room[model_names[ii]]))

        accuracy_by_frame_2spk_by_room[model_names[ii]].append(np.mean(accuracy_by_frame_2spk[model_names[ii]],axis=(1,2)))
        accuracy_by_frame_2spk_mean[model_names[ii]].append(np.mean(accuracy_by_frame_2spk_by_room[model_names[ii]]))

        accuracy_by_frame_1spk_by_room_medfilt[model_names[ii]].append(np.mean(accuracy_by_frame_1spk_medfilt[model_names[ii]],axis=(1,2)))
        accuracy_by_frame_1spk_mean_medfilt[model_names[ii]].append(np.mean(accuracy_by_frame_1spk_by_room_medfilt[model_names[ii]]))

        accuracy_by_frame_2spk_by_room_medfilt[model_names[ii]].append(np.mean(accuracy_by_frame_2spk_medfilt[model_names[ii]],axis=(1,2)))
        accuracy_by_frame_2spk_mean_medfilt[model_names[ii]].append(np.mean(accuracy_by_frame_2spk_by_room_medfilt[model_names[ii]]))

        count_spk_room1_sum[model_names[ii]].append(np.sum(count_spk[:,0,:,:],axis=(1,2)))
        count_spk_room2[model_names[ii]].append(count_spk[:,1,:,:])
        count_spk_room2_sum[model_names[ii]].append(np.sum(count_spk[:,1,:,:],axis=(1,2)))
        """

if __name__ == "__main__":
    args = parseArgs()
    main(args)