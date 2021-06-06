# Author: Lior Frenkel

import argparse
import random
from random import seed
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd
import os
from os import listdir
from scipy.io import wavfile, savemat
#from scipy.signal import stft

import nprirgen

parser = argparse.ArgumentParser(description='Localization')
parser.add_argument('--max_speakers', type=int, default=4, metavar='N',
                    help='The max number of speakers (default: 4)')
parser.add_argument('--num_results', type=int, default=5, metavar='NR',
                    help='Number of results (default: 5)')
parser.add_argument('--start_results', type=int, default=0, metavar='SR',
                    help='The starting index of results (default: 0)')
parser.add_argument('--mode', type=str, default='train',
                    help='Mode, i.e., train, val or test (default: train)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed (default: 1)')
parser.add_argument('--n_mics', type=int, default=4, metavar='NM',
                    help='Number of microphones (default: 4)')
parser.add_argument('--s_path', type=str, default='/mnt/dsi_vol1/users/frenkel2/data/localization/WSJ1',
                    help='Speakers .wav files directory (default: /mnt/dsi_vol1/users/frenkel2/data/localization/WSJ1)')
parser.add_argument('--n_path', type=str, default='/mnt/dsi_vol1/users/frenkel2/data/localization/wham_noise',
                    help='Noise .wav files directory (default: /mnt/dsi_vol1/users/frenkel2/data/localization/wham_noise)')
parser.add_argument('--results_path', type=str, default='/mnt/dsi_vol1/users/frenkel2/data/localization/results/new/two_speakers_20000_samples',
                    help='results directory (default: /mnt/dsi_vol1/users/frenkel2/data/localization/results/new/two_speakers_20000_samples)')
parser.add_argument('--n_speak', type=int, default=2, metavar='NS',
                    help='Number of speakers if not random (default: 2)')
parser.add_argument('--not_rand_speak', action='store_true',
                     help='if specified, num of speakers is not random')
parser.add_argument('--add_noise', action='store_true',
                     help='if specified, adding a random noise to speakers')
parser.add_argument('--create_maps', action='store_true',
                     help='if specified, creating a freq-time map for each scenario')
parser.add_argument('--no_save_wav', action='store_true',
                     help='if specified, not saving wav files results')
args = parser.parse_args()

seed(args.seed)


class Room:
    def __init__(self):
        super().__init__()

        min_w = 4
        max_w = 8
        min_l = 4
        max_l = 8
        min_h = 2.5
        max_h = 3
        min_rt60 = 0.13
        max_rt60 = 0.8

        self.room_width = np.random.uniform(min_w, max_w)
        self.room_length = np.random.uniform(min_l, max_l)
        self.room_height = np.random.uniform(min_h, max_h)
        self.rt60 = np.random.uniform(min_rt60, max_rt60)

        print('Room shape is {0:.2f}x{1:.2f}x{2:.2f}'.format(self.room_width, self.room_length, self.room_height))

    def get_dims(self):
        return round(self.room_width, 2), round(self.room_length, 2), round(self.room_height, 2), self.rt60


class Array:
    def __init__(self, width, length, array_area_w=0.5, array_area_l=0.5, n_mic=4,
                 mic_dists=[0.08, 0.08, 0.08], res=5):
        super().__init__()

        min_array_z = 1
        max_array_z = 1.7

        self.array_x = np.random.uniform(0.5*width - array_area_w, 0.5*width + array_area_w)
        self.array_y = np.random.uniform(0.5*length - array_area_l, 0.5*length + array_area_l)
        self.array_z = round(np.random.uniform(min_array_z, max_array_z), 2)
        self.n_mic = n_mic
        self.mic_dists = mic_dists
        
        theta_opt = np.arange(0, 180, res)
        theta_inx = np.random.randint(len(theta_opt))
        self.array_theta = theta_opt[theta_inx]

        print('Array center was located in ({0:.2f},{1:.2f},{2:.2f}) with theta = {3}'
              .format(self.array_x, self.array_y, self.array_z, self.array_theta))

    def get_array_loc(self):
        receivers = []
        if not self.mic_dists:  # One mic
            receivers.append([round(self.array_x, 2), round(self.array_y, 2), self.array_z])
        else:
            radius1 = sum(self.mic_dists) / 2
            if self.array_theta <= 90:
                mic_x1 = self.array_x - radius1 * math.cos(math.radians(self.array_theta))
                mic_y1 = self.array_y - radius1 * math.sin(math.radians(self.array_theta))
                receivers.append([round(mic_x1, 2), round(mic_y1, 2), self.array_z])
                for dist in self.mic_dists:
                    mic_x = mic_x1 + dist * math.cos(math.radians(self.array_theta))
                    mic_y = mic_y1 + dist * math.sin(math.radians(self.array_theta))
                    receivers.append([round(mic_x, 2), round(mic_y, 2), self.array_z])
                    mic_x1 = mic_x
                    mic_y1 = mic_y
            else:
                mic_x1 = self.array_x - radius1 * math.cos(math.radians(180 - self.array_theta))
                mic_y1 = self.array_y + radius1 * math.sin(math.radians(180 - self.array_theta))
                receivers.append([round(mic_x1, 2), round(mic_y1, 2), self.array_z])
                for dist in self.mic_dists:
                    mic_x = mic_x1 + dist * math.cos(math.radians(180 - self.array_theta))
                    mic_y = mic_y1 - dist * math.sin(math.radians(180 - self.array_theta))
                    receivers.append([round(mic_x, 2), round(mic_y, 2), self.array_z])
                    mic_x1 = mic_x
                    mic_y1 = mic_y

        return self.array_x, self.array_y, self.array_z, self.array_theta, receivers


class Speakers:
    def __init__(self, args, max_speakers, width, length, height, array_x, array_y, array_z, array_theta,
                 rt60, n_mic=4, mic_dists=[0.08, 0.08, 0.08], limit=0.5):
        super().__init__()

        self.args = args
        if args.not_rand_speak:
            self.N = args.n_speak
        else:
            self.N = np.random.randint(1, max_speakers +1)
        if args.add_noise:
            self.N += 1
        self.width = width
        self.length = length
        self.height = height
        self.rt60 = rt60
        self.array_x = array_x
        self.array_y = array_y
        self.array_z = array_z
        self.array_theta = array_theta
        self.n_mic = n_mic
        self.mic_dists = mic_dists
        self.limit = limit

        if args.add_noise:
            print('Number of speakers is: ', self.N - 1)
        else:
            print('Number of speakers is: ', self.N)

    def find_r_theta(self, theta_opt):
        theta_inx = np.random.randint(len(theta_opt))
        speaker_theta = theta_opt[theta_inx]
        if speaker_theta < 180 and speaker_theta != 0 and speaker_theta != 90:
            y_limit = self.length - self.limit - self.array_y
            x_max = y_limit / (math.tan(math.radians(speaker_theta)))
            if speaker_theta < 90:
                x_limit = self.width - self.limit - self.array_x
            elif speaker_theta > 90:
                x_limit = -(self.array_x - self.limit)
            y_max = math.tan(math.radians(speaker_theta)) * x_limit
            if speaker_theta < 90:
                max_speaker_x = min([x_max, x_limit])
            elif speaker_theta > 90:
                max_speaker_x = max([x_max, x_limit])
            max_speaker_y = min([y_max, y_limit])
            max_speaker_r = math.sqrt(pow(max_speaker_x, 2) + pow(max_speaker_y, 2))
        elif speaker_theta > 180 and speaker_theta != 270:
            y_limit = -(self.array_y - self.limit)
            x_max = y_limit / (math.atan(math.radians(speaker_theta - 180)))
            if speaker_theta < 270:
                x_limit = -(self.array_x - self.limit)
            elif speaker_theta > 270:
                x_limit = self.width - self.limit - self.array_x
            y_min = math.tan(math.radians(speaker_theta - 180)) * x_limit
            if speaker_theta < 270:
                max_speaker_x = max([x_max, x_limit])
            elif speaker_theta > 270:
                max_speaker_x = min([x_max, x_limit])
            max_speaker_y = max([y_min, y_limit])
            max_speaker_r = math.sqrt(pow(max_speaker_x, 2) + pow(max_speaker_y, 2))
        elif speaker_theta == 0:
            max_speaker_r = self.width - self.limit - self.array_x
        elif speaker_theta == 90:
            max_speaker_r = self.length - self.limit - self.array_y
        elif speaker_theta == 180:
            max_speaker_r = self.array_x - self.limit
        elif speaker_theta == 270:
            max_speaker_r = self.array_y - self.limit

        mic2speaker_min = 1
        speaker_r = np.random.uniform(mic2speaker_min, max_speaker_r)

        if 0 < self.array_theta <= 90:
            speaker_x = self.array_x + speaker_r * math.cos(math.radians(speaker_theta))
            speaker_y = self.array_y + speaker_r * math.sin(math.radians(speaker_theta))
        elif 90 < self.array_theta <= 180:
            speaker_x = self.array_x - speaker_r * math.cos(math.radians(180 - speaker_theta))
            speaker_y = self.array_y + speaker_r * math.sin(math.radians(180 - speaker_theta))
        elif 180 < self.array_theta <= 270:
            speaker_x = self.array_x - speaker_r * math.cos(math.radians(speaker_theta))
            speaker_y = self.array_y - speaker_r * math.sin(math.radians(speaker_theta))
        else:
            speaker_x = self.array_x + speaker_r * math.cos(math.radians(180 - speaker_theta))
            speaker_y = self.array_y - speaker_r * math.sin(math.radians(180 - speaker_theta))
        speaker_z = round(np.random.uniform(1.3, 1.9), 2)
        #speaker_z = 1.6

        if self.array_theta <= 90:
            if speaker_theta >= self.array_theta:
                speaker_theta_array = speaker_theta - self.array_theta
                """
                if speaker_theta < 180 + self.array_theta:
                    speaker_theta_array = speaker_theta - self.array_theta
                else:
                    speaker_theta_array = speaker_theta - (180 + self.array_theta)
                """
            else:
                speaker_theta_array = 360 - (self.array_theta - speaker_theta)
                #speaker_theta_array = 180 - (self.array_theta - speaker_theta)
        else:
            if speaker_theta >= self.array_theta + 180:
                speaker_theta_array = speaker_theta - self.array_theta - 180
            else:
                speaker_theta_array = speaker_theta - self.array_theta + 180
                """
                if speaker_theta > self.array_theta:
                    speaker_theta_array = speaker_theta - self.array_theta
                else:
                    speaker_theta_array = speaker_theta - self.array_theta + 180  
                """     

        return round(speaker_r, 2), speaker_theta, speaker_theta_array,\
               round(speaker_x, 2), round(speaker_y, 2), speaker_z

    def get_first_speaker(self, res=5):
        theta_opt = np.arange(0, 360, res)
        speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z = self.find_r_theta(theta_opt)
        print('First speaker angle from array is: ', s_theta_array)
        print('First speaker radius from array is: {0:.2f}'.format(speaker_r))

        return speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z

    def get_speaker(self, angles, i, islast, res=5, speaker_dist=20):
        theta_opt = np.arange(0, 360, res)
        for a in angles:
            del_inx = np.where(((theta_opt < a + speaker_dist) & (theta_opt > a - speaker_dist))
                               | (theta_opt > 360 + a - speaker_dist) | (theta_opt < a - 360 - speaker_dist))
            theta_opt = np.delete(theta_opt, del_inx)
        speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z = self.find_r_theta(theta_opt)
        if (not self.args.add_noise) or (self.args.add_noise and not islast):
            print('Speaker {0} angle from array is: {1}'.format(i, s_theta_array))
            print('Speaker {0} radius from array is: {1:.2f}'.format(i, speaker_r))

        return speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z

    def get_speakers(self, mics_loc, i):
        r1, theta1, theta1_array, x1, y1, z1 = self.get_first_speaker()
        speakers_loc = []
        thetas = []
        thetas_array = []
        dists = []
        speakers_loc.append([x1, y1, z1])
        thetas.append(theta1)
        thetas_array.append(theta1_array)
        dists.append(r1)
        is_last = False
        for s in range(1, self.N):
            if s == self.N - 1:
                is_last = True
            r, theta, theta_array, x, y, z = self.get_speaker(thetas, s+1, is_last)
            thetas.append(theta)
            thetas_array.append(theta_array)
            dists.append(r)
            speakers_loc.append([x, y, z])
        if self.args.add_noise:
            print('Speakers angles from array are: ', thetas_array[:-1])
            print('Speakers distances from array are: ', dists[:-1])
        else:
            print('Speakers angles from array are: ', thetas_array)
            print('Speakers distances from array are: ', dists)
        h_list = self.get_speakers_h(speakers_loc, mics_loc)
        speakers_id, speakers_path = self.get_speakers_id()
        speakers_wav_files = self.choose_wav(speakers_path)
        speakers_conv, _ = self.conv_wav_h(speakers_wav_files, h_list, i, thetas_array)     

        return self.N, dists, thetas_array, speakers_loc, speakers_id, h_list, speakers_conv

    def get_speaker_h(self, speaker_x, speaker_y, speaker_z, receivers):
        room_measures = [self.width, self.length, self.height]
        source_position = [speaker_x, speaker_y, speaker_z]
        fs = 16000
        n = 2048
        h, _, _ = nprirgen.np_generateRir(room_measures, source_position, receivers, reverbTime=self.rt60,
                                          fs=fs, orientation=[self.array_theta, .0], nSamples=n)

        return h

    def get_speakers_h(self, speakers_loc, mics_loc):
        speakers_h = []
        for inx, speaker in enumerate(speakers_loc):
            h = self.get_speaker_h(speaker[0], speaker[1], speaker[2], mics_loc)
            speakers_h.append(h)
            """
            print('h of speaker number {0}: '.format(inx + 1), h)
            """
        return speakers_h
    
    def get_speakers_id(self):
        if self.args.mode == 'train' or self.args.mode == 'val':
            PATH = self.args.s_path + '/Train'
            ids = listdir(PATH)
            ids.remove('Train.zip')
        elif self.args.mode == 'test':
            PATH = self.args.s_path + '/Test'
            ids = listdir(PATH)
            ids.remove('female')
            ids.remove('male')
        else:
            raise Exception("Chosen mode is not valid!!")
        if self.args.add_noise:
            id_list = random.choices(ids, k=self.N - 1)
        else:
            id_list = random.choices(ids, k=self.N)
        id_path = [os.path.join(PATH, id) for id in id_list]
        if self.args.add_noise:
            noise = Noise(self.args)
            id_path.append(noise.choose_noise_path())
        
        return id_list, id_path
    
    def choose_wav(self, id_path):
        wav_lists = []
        for path in id_path:
            wav_lists.append([os.path.join(path, wav_file) for wav_file in listdir(path)])
        wav_lists1 = []
        if self.args.add_noise:
            for wav_list in wav_lists[:-1]:
                wav_lists1.append([path for path in wav_list if os.path.splitext(path)[1] == '.wav'])
            wav_lists1.append(wav_lists[-1])
        else:
            for wav_list in wav_lists:
                wav_lists1.append([path for path in wav_list if os.path.splitext(path)[1] == '.wav'])
        wav_files = [random.choice(wav_list) for wav_list in wav_lists1]

        return wav_files
    
    def conv_wav_h(self, wav_files, h_list, i, thetas_array):
        THRESHOLD = 40
        #FRAMES_PER_SAMPLE = 256
        GLOBAL_MEAN = 44
        GLOBAL_STD = 15.5
        noise_angle = -1
        speakers_conv = []
        fss = []
        if self.args.add_noise:
            noise_file = wav_files[-1]
            noise_h = h_list[-1]
            wav_files = wav_files[:-1]
            h_list = h_list[:-1]
        for wave_file, h in zip(wav_files, h_list):
            fs, wave = wavfile.read(wave_file)
            fss.append(fs)
            wave = np.copy(wave)

            # Normalizing wave data
            wave = wave / 1.1
            wave = wave / np.max(np.absolute(wave))
            speaker_conv = []
            for s_mic_h in h:
                wav_h_conv = np.convolve(wave, s_mic_h)
                speaker_conv.append(np.expand_dims(wav_h_conv, axis=0))
            speaker_conv_np = np.concatenate(speaker_conv)
            speakers_conv.append(speaker_conv_np)

        
        if self.args.add_noise:
            n_fs, noise = wavfile.read(noise_file)
            fss = [n_fs] + fss
            noise = np.copy(noise[:, 0])  # Choose one noise channel

            noise_conv = []
            for n_mic_h in noise_h:
                noise_h_conv = np.convolve(noise, n_mic_h)
                noise_conv.append(np.expand_dims(noise_h_conv, axis=0))
            noise_conv_np = np.concatenate(noise_conv)
            speakers_conv.append(noise_conv_np)
        
        len_list = [arr.shape[1] for arr in speakers_conv]
        min_len = min(len_list)
        speakers_conv_cut = [arr[:,:min_len] for arr in speakers_conv]
        if self.args.create_maps:
            mics_list = []
            mics_list_soumitro = []
            for mic in range(self.args.n_mics):
                #if mic + 1 == math.ceil(self.args.n_mics / 2):
                stft_list = []
                stft_list_soumitro = []
                stfts = []
                for s_inx, s_conv in enumerate(speakers_conv_cut):
                    S, speech_1_spec, speech_1_spec_log = signal_pre_process(s_conv[mic, :], nfft=512, overlap=0.75)
                    if self.args.mode == 'train':
                        stft_list.append(speech_1_spec)
                        stfts.append(S)
                    elif self.args.mode == 'test':
                        stft_list.append(speech_1_spec_log)
                        _, _, speech_1_spec_log_soumitro = signal_pre_process(s_conv[mic, :], nfft=512, overlap=0.5)
                        stft_list_soumitro.append(speech_1_spec_log_soumitro)
                mics_list.append(stft_list)
                if self.args.mode == 'test':
                    mics_list_soumitro.append(stft_list_soumitro)
        if self.args.add_noise:
            noise_conv_cut = speakers_conv_cut[-1]
            speakers_conv_np = sum(speakers_conv_cut[:-1])
            speakers_conv_np = Noise(self.args).get_mixed(speakers_conv_np, noise_conv_cut)
        else:
            speakers_conv_np = sum(speakers_conv_cut)

        # Normalizing result to between -1 and 1
        max_conv_val = np.max(speakers_conv_np)
        min_conv_val = np.min(speakers_conv_np)
        speakers_conv_np = np.transpose(2 / (max_conv_val - min_conv_val) *
                                        (speakers_conv_np - max_conv_val) + 1)

        if self.args.create_maps:
            mixed_mics_list = []
            mixed_mics_list2 = []
            TrainingData = {}
            for mic in range(speakers_conv_np.shape[1]):
                if ((mic + 1 == math.ceil(self.args.n_mics / 2)) and self.args.mode == 'test') or self.args.mode == 'train':
                    speech_mix_spec0, _, speech_mix_spec_log = signal_pre_process(speakers_conv_np[:, mic], nfft=512, overlap=0.75)
                    if self.args.mode == 'train':
                        speech_mix_spec = (speech_mix_spec_log - GLOBAL_MEAN) / GLOBAL_STD
                        mixed_mics_list.append(speech_mix_spec)
                    elif self.args.mode == 'test':
                        mixed_mics_list.append(speech_mix_spec_log)
                    max_mag = np.max(speech_mix_spec_log)
                    speech_VAD = speech_mix_spec_log > (max_mag - THRESHOLD)
                    s_masks = []
                    for s in mics_list[mic]:
                        speech_1_mask = np.zeros(s.shape)
                        speech_1_mask[s == np.maximum.reduce(mics_list[mic])] = 1
                        speech_1_mask = speech_1_mask*speech_VAD
                        s_masks.append(speech_1_mask)
                    if self.args.mode == 'test':
                        _, _, speech_mix_spec_log_soumitro = signal_pre_process(speakers_conv_np[:, mic], nfft=512, overlap=0.5)
                        mixed_mics_list2.append(speech_mix_spec_log_soumitro)
                        max_mag_soumitro = np.max(speech_mix_spec_log_soumitro)
                        speech_VAD_soumitro = speech_mix_spec_log_soumitro > (max_mag_soumitro - THRESHOLD)
                        s_masks_soumitro = []
                        for s in mics_list_soumitro[mic]:
                            speech_1_mask_soumitro = np.zeros(s.shape)
                            speech_1_mask_soumitro[s == np.maximum.reduce(mics_list_soumitro[mic])] = 1
                            speech_1_mask_soumitro = speech_1_mask_soumitro*speech_VAD_soumitro
                            s_masks_soumitro.append(speech_1_mask_soumitro)
                    if self.args.mode == 'train':
                        #k = 0
                        curr_inx = 0
                        len_spec = stfts[0].shape[1]
                        #while (k + FRAMES_PER_SAMPLE) < len_spec:
                        if mic == 0:
                            """
                            TrainingData[curr_inx] = {}
                            TrainingData[curr_inx]['labelMat'] = np.zeros((stfts[0].shape[0], FRAMES_PER_SAMPLE, self.args.n_mics))
                            TrainingData[curr_inx]['sampleSTFT'] = np.zeros((stfts[0].shape[0], FRAMES_PER_SAMPLE, self.args.n_mics), dtype=complex)
                            TrainingData[curr_inx]['VAD_whole'] = np.zeros((stfts[0].shape[0], FRAMES_PER_SAMPLE, self.args.n_mics))
                            """
                            TrainingData[curr_inx] = {}
                            TrainingData[curr_inx]['labelMat'] = np.zeros((stfts[0].shape[0], len_spec, self.args.n_mics))
                            TrainingData[curr_inx]['sampleSTFT'] = np.zeros((stfts[0].shape[0], len_spec, self.args.n_mics), dtype=complex)
                            TrainingData[curr_inx]['VAD_whole'] = np.zeros((stfts[0].shape[0], len_spec, self.args.n_mics))
                        samples = []
                        for s in mics_list[mic]: 
                            #samples.append(s[:, k:k + FRAMES_PER_SAMPLE])
                            samples.append(s)
                        label_mat = np.zeros(samples[0].shape)
                        for s_inx, s_angle in enumerate(thetas_array): 
                            #label_mat[s_masks[s_inx][:, k:k + FRAMES_PER_SAMPLE] == 1] = s_angle
                            label_mat[s_masks[s_inx] == 1] = s_angle
                        #label_mat[speech_VAD[:, k:k + FRAMES_PER_SAMPLE] == 0] = noise_angle
                        label_mat[speech_VAD == 0] = noise_angle
                        """
                        sample_mix = speech_mix_spec[:, k:k + FRAMES_PER_SAMPLE - 1]
                        Y = []
                        for s in mics_list[mic]: 
                            Y.append(np.expand_dims((s == np.maximum.reduce(samples)).flatten(), axis=1))
                        Y = np.concatenate(Y, axis=1)
                        """
                        TrainingData[curr_inx]['labelMat'][:, :, mic] = label_mat
                        #TrainingData[curr_inx]['sampleSTFT'][:, :, mic] = speech_mix_spec0[:, k:k + FRAMES_PER_SAMPLE]
                        TrainingData[curr_inx]['sampleSTFT'][:, :, mic] = speech_mix_spec0
                        #TrainingData[curr_inx]['VAD_whole'][:, :, mic] = speech_VAD[:, k:k + FRAMES_PER_SAMPLE]
                        TrainingData[curr_inx]['VAD_whole'][:, :, mic] = speech_VAD
                        #k += FRAMES_PER_SAMPLE
                        curr_inx += 1
                    elif self.args.mode == 'test':
                        label_mat = np.zeros(s_masks[0].shape)
                        label_mat_soumitro = np.zeros(s_masks_soumitro[0].shape)
                        for s_inx, s_angle in enumerate(thetas_array): 
                            label_mat[s_masks[s_inx] == 1] = s_angle
                            label_mat_soumitro[s_masks_soumitro[s_inx] == 1] = s_angle
                        label_mat[speech_VAD == 0] = noise_angle
                        label_mat_soumitro[speech_VAD_soumitro == 0] = noise_angle
                    if self.args.mode == 'train' and mic == self.args.n_mics - 1:
                        for kk in range(curr_inx):
                            if self.args.add_noise:
                                #mat_path = '{0}/conv_{1}_{2}_{3}.mat'.format(self.args.results_path, i + 1, self.N - 1, kk)
                                mat_path = '{0}/conv_{1}_{2}.mat'.format(self.args.results_path, i + 1, self.N - 1)
                            else:
                                #mat_path = '{0}/conv_{1}_{2}_{3}.mat'.format(self.args.results_path, i + 1, self.N, kk)
                                mat_path = '{0}/conv_{1}_{2}.mat'.format(self.args.results_path, i + 1, self.N)
                            savemat(mat_path, {'x_train':TrainingData[kk]['sampleSTFT'], 'y_train':TrainingData[kk]['labelMat'], 'VAD':TrainingData[kk]['VAD_whole']})
                    elif self.args.mode == 'test':
                        if self.args.add_noise:
                            mat_path = '{0}/conv_{1}_{2}_trueLABEL.mat'.format(self.args.results_path, i + 1, self.N - 1)
                            mat_path_soumitro = '{0}/conv_{1}_{2}_trueLABELsoumitro.mat'.format(self.args.results_path, i + 1, self.N - 1)
                        else:
                            mat_path = '{0}/conv_{1}_{2}_trueLABEL.mat'.format(self.args.results_path, i + 1, self.N)
                            mat_path_soumitro = '{0}/conv_{1}_{2}_trueLABELsoumitro.mat'.format(self.args.results_path, i + 1, self.N)
                        savemat(mat_path, {'label_mat':label_mat})
                        savemat(mat_path_soumitro, {'label_mat_soumitro':label_mat_soumitro})


        if self.args.add_noise:
            conv_path = '{0}/noisy/conv_{1}_{2}.wav'.format(self.args.results_path, i + 1, self.N - 1)
        else:
            conv_path = '{0}/conv_{1}_{2}.wav'.format(self.args.results_path, i + 1, self.N)
        if not self.args.no_save_wav:    
            wavfile.write(conv_path, fs, speakers_conv_np)
        
        return speakers_conv_np, conv_path


class Noise:
    def __init__(self, args):
        super().__init__()

        #self.snr = np.random.uniform(5, 15)
        self.snr = 15
        self.args = args

    def choose_noise_path(self):
        if self.args.mode == 'train':
            path = self.args.n_path + '/tr'
        elif self.args.mode == 'val':
            path = self.args.n_path + '/cv'
        elif self.args.mode == 'test':
            path = self.args.n_path + '/tt'
        else:
            raise Exception("Chosen mode is not valid!!")

        return path

    def get_mixed(self, mixed, noise):
        noise_gain = np.sqrt(10 ** (-self.snr / 10) * np.power(np.std(mixed), 2) / np.power(np.std(noise), 2))
        noise = noise_gain * noise
        mixed = mixed + noise
        
        return mixed


def signal_pre_process(s, min_amp=10000, amp_fac=10000, nfft=512, overlap=0.75):
    S = stft(s, nfft=nfft, overlap=overlap).T
    speech_spec = np.absolute(S)
    speech_spec = np.maximum(speech_spec, np.max(speech_spec)/min_amp)
    speech_spec_log = 20*np.log10(speech_spec*amp_fac)
    
    return S, speech_spec, speech_spec_log

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

def plot_room(args, width, length, height, speakers_loc, mic_xy, scenario):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.title('Scenario {0}'.format(scenario))
    ax.set_xlim(0, width)
    ax.set_ylim(0, length)
    ax.set_zlim(0, height)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    speakers_loc_np = np.array(speakers_loc)
    mic_xy_np = np.array(mic_xy)
    speakers_x = speakers_loc_np[:, 0]
    speakers_y = speakers_loc_np[:, 1]
    speakers_z = speakers_loc_np[:, 2]
    mics_x = mic_xy_np[:, 0]
    mics_y = mic_xy_np[:, 1]
    mics_z = mic_xy_np[:, 2]
    ax.scatter(speakers_x, speakers_y, speakers_z, marker='o', s=20, c='b', depthshade=True)
    ax.scatter(mics_x, mics_y, mics_z, marker='x', s=20, c='r', depthshade=True)
    plt.savefig('{0}/scenario number {1}.png'.format(args.results_path, scenario + 1))


def get_scenario_data(args, scenario):
    room = Room()
    w, l, h, rt60 = room.get_dims()
    array = Array(w, l)
    array_x, array_y, array_z, array_theta, mics_xy = array.get_array_loc()
    max_speakers = args.max_speakers
    speakers = Speakers(args, max_speakers, w, l, h, array_x, array_y, array_z, array_theta, rt60)
    _, dists, thetas, speakers_xy, speakers_id, h_list, conv = speakers.get_speakers(mics_xy, scenario)
    """
    plot_room(args, w, l, h, speakers_xy, mics_xy, scenario)
    """

    return w, l, h, rt60, array_theta, mics_xy, speakers_xy, dists, thetas, speakers_id, h_list, conv


def create_csv_results(args):
    with open('{0}/localization_data.csv'.format(args.results_path), 'w', newline='') as file:
        first_row = ['scenario', 'room_x', 'room_y', 'room_z', 'rt60']
        first_row.append('num_mics')
        dims = ['x', 'y', 'z']
        for mic in range(args.n_mics):
            for dim in dims:
                first_row.append('mic{0}_{1}'.format(mic + 1, dim))
        first_row.append('array_theta')
        first_row.append('num_speakers')
        for s in range(args.max_speakers):
            first_row.append('speaker{0}_id'.format(s + 1))
            first_row.append('speaker{0}_radius'.format(s + 1))
            first_row.append('speaker{0}_theta'.format(s + 1))
        for s in range(args.max_speakers):
            for dim in dims:
                first_row.append('speaker{0}_{1}'.format(s + 1, dim))
        writer = csv.DictWriter(file, fieldnames=first_row)
        writer.writeheader()

        for result in range(args.start_results, args.num_results):
            w, l, h, rt60, array_theta, mics_xy, speakers_xy, dists, thetas, \
            speakers_id, _, _ = get_scenario_data(args, result)
            row_dict = {'scenario': result + 1, 'room_x': w, 'room_y': l, 'room_z': h, 'rt60': round(rt60, 2)}
            mics_xy_np = np.array(mics_xy)
            row_dict['num_mics'] = args.n_mics
            for mic in range(args.n_mics):
                for dim_inx, dim in enumerate(dims):
                    if mics_xy_np.ndim == 1:
                        row_dict['mic{0}_{1}'.format(mic + 1, dim)] = mics_xy_np[dim_inx]
                    else:
                        row_dict['mic{0}_{1}'.format(mic + 1, dim)] = mics_xy_np[mic][dim_inx]
            row_dict['array_theta'] = array_theta
            speakers_xy_np = np.array(speakers_xy)
            if args.add_noise:
                speakers_xy = speakers_xy[:-1]
            row_dict['num_speakers'] = len(speakers_xy)
            for s in range(len(speakers_xy)):
                row_dict['speaker{0}_id'.format(s + 1)] = speakers_id[s]
                row_dict['speaker{0}_radius'.format(s + 1)] = dists[s]
                row_dict['speaker{0}_theta'.format(s + 1)] = thetas[s]
            for s in range(len(speakers_xy)):
                for dim_inx, dim in enumerate(dims):
                    if speakers_xy_np.ndim == 1:
                        row_dict['speaker{0}_{1}'.format(s + 1, dim)] = speakers_xy_np[dim_inx]
                    else:
                        row_dict['speaker{0}_{1}'.format(s + 1, dim)] = speakers_xy_np[s][dim_inx]
            writer.writerow(row_dict)
        file.close()


if __name__ == "__main__":
    create_csv_results(args)

