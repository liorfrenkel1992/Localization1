import os
import sys
import gc
import argparse

import numpy as np
import hdf5storage
from scipy.io import savemat


eps = sys.float_info.epsilon

parser = argparse.ArgumentParser(description='Localization')
parser.add_argument('--raw_path', type=str, default='/mnt/dsi_vol1/users/frenkel2/data/localization/results/new/two_speakers_20000_samples',
                    help='Speakers .mat files directory (default: /mnt/dsi_vol1/users/frenkel2/data/localization/results/new/two_speakers_20000_samples)')
parser.add_argument('--processed_path', type=str, default='/mnt/dsi_vol1/users/frenkel2/data/localization/Git/localization1/data/processed',
                    help='Speakers .mat files directory (default: /mnt/dsi_vol1/users/frenkel2/data/localization/Git/localization1/data/processed)')
parser.add_argument('--frames_per_sample', type=int, default=256, metavar='FPS',
                    help='Frames per sample (default: 256)')
parser.add_argument('--spec_size', type=int, default=256, metavar='SPS',
                    help='Size of spectrum (default: 256)')
parser.add_argument('--frame_size', type=int, default=256, metavar='FS',
                    help='Size of frame (default: 256)')
parser.add_argument('--spec_var', type=int, default=3, metavar='SV',
                    help='Spectrum fixed variance for std normalization (default: 3)')
parser.add_argument('--n_class', type=int, default=37, metavar='NC',
                    help='Number of possible angles (default: 37)')
parser.add_argument('--n_mics', type=int, default=4, metavar='NM',
                    help='Number of microphones (default: 4)')
parser.add_argument('--ref_channel', type=int, default=2, metavar='RM',
                    help='Reference microphone (default: 2)')
parser.add_argument('--eps', type=float, default=eps, metavar='EPS',
                    help='Size of eps for not dividing by zero / log a small number')
parser.add_argument('-std_norm', action="store_true", dest='std_norm',
                        help='Normalize input by std')
args = parser.parse_args()


def preprocess(args):
    imgs = os.listdir(args.raw_path)
    for img in imgs:
        inx = img.split('_')[1]
        print('Processing img number {0}'.format(inx))
        n_speak = img.split('_')[2].split('.')[0]
        img_path = os.path.join(args.raw_path, img)
        mat = hdf5storage.loadmat(img_path)
        x = mat['x_train']
        y = mat['y_train']
        if x.shape[1] > args.frames_per_sample + 1:
            start_point = np.random.randint(x.shape[1] - args.frames_per_sample + 1)
            x = x[:, start_point:start_point + args.frames_per_sample, :]
            y = y[:, start_point:start_point + args.frames_per_sample, :]
        else:  # Freq dimension is smaller than frames_per_samle
            x_temp = x
            y_temp = y
            x = np.zeros((x_temp.shape[0], args.frames_per_sample, x_temp.shape[2]), dtype=complex)
            y = np.zeros((y_temp.shape[0], args.frames_per_sample, y_temp.shape[2]))
            x[:, :x_temp.shape[1], :] = x_temp
            y[:, :y_temp.shape[1], :] = y_temp
        y = np.where(y > 180, 360 - y, y)
        y[y == -1] = 270
        
        # %% CREATE TRAINING LABELS

        y_sum = np.sum(y, axis=2)
        y_sum_avg = y_sum / args.n_mics
        VAD_new = y_sum > 270 * (args.n_mics - 1)
        VAD_new = 1 - VAD_new
        y_labels = np.unique(y)
        y = np.zeros(np.shape(y_sum))
        
        if y_labels.shape[0] == 2:
            y[VAD_new == 1] = y_labels[0]
        elif y_labels.shape[0] == 3:
            y[(abs(y_sum_avg - y_labels[0]) < abs(y_sum_avg - y_labels[1])) & VAD_new == 1] = y_labels[0]
            y[(abs(y_sum_avg - y_labels[1]) <= abs(y_sum_avg - y_labels[0])) & VAD_new == 1] = y_labels[1]
        elif y_labels.shape[0] > 3:
            y[(abs(y_sum_avg - y_labels[0]) < abs(y_sum_avg - y_labels[1])) & (abs(y_sum_avg - y_labels[0]) < abs(y_sum_avg - y_labels[2]))
                & VAD_new == 1] = y_labels[0]
            y[(abs(y_sum_avg - y_labels[1]) <= abs(y_sum_avg - y_labels[0])) & (abs(y_sum_avg - y_labels[1]) < abs(y_sum_avg - y_labels[2]))
                & VAD_new == 1] = y_labels[1]
            y[(abs(y_sum_avg - y_labels[2]) <= abs(y_sum_avg - y_labels[0])) & (abs(y_sum_avg - y_labels[2]) < abs(y_sum_avg - y_labels[1]))
                & VAD_new == 1] = y_labels[2]
        y[VAD_new == 0] = 270

        del y_sum, y_sum_avg, y_labels, VAD_new
        gc.collect()
        
        y = y[:args.spec_size, :args.frame_size]
        y1 = y.astype(int)

        
        #y = y1.reshape(self.spec_size, self.frame_size, 1)

        for ii in range(args.n_class):
            y[y1 == ii * 5] = ii

        #y[y == 270] = 0
        

        # %% CREATE TRAINING DATA

        x = x + args.eps
        x_spec = np.log(args.eps + np.abs(x[0:args.spec_size, :, args.ref_channel]))
        
        x_rtf = x / np.expand_dims(x[:, :, args.ref_channel], 2)
        x_rtf = np.concatenate((x_rtf[:, :, :args.ref_channel], x_rtf[:, :, args.ref_channel + 1:]), axis=-1)
        x_rtf = x_rtf[:args.spec_size, :, :]

        
        # if args.std_norm:
        #     # x_rtf_std = np.expand_dims(np.expand_dims(x_rtf.std(axis=(0, 1)), 0), 0)
        #     # x_rtf_std = np.tile(x_rtf_std, (args.spec_size, args.frame_size, 1))
        #     x_rtf_std = x_rtf.std(axis=(0, 1))[args.ref_channel - 1]  # Divide all by 1 mic std
        #     x_rtf = x_rtf * ((np.sqrt(1) / (eps + x_rtf_std)))
            
        
        x_real = np.real(x_rtf)
        x_image = np.imag(x_rtf)
        
        ### Make rtf values in range [0,1]
        global_min = min(x_real.min(), x_image.min())
        global_max = max(x_real.max(), x_image.max())
        x_real = (x_real - global_min) / (global_max - global_min)
        x_image = (x_image - global_min) / (global_max - global_min)
        
        del x_rtf
        gc.collect()

        # if args.std_norm:
        #     ## varaince 1 for every spectrogram
        #     # x_spec_std = np.expand_dims(np.expand_dims(x_spec.std(axis=(0, 1)), 0), 0)
        #     # x_spec_std = np.tile(x_spec_std, (args.spec_size, args.frame_size, 1))
        #     x_spec_std = x_spec.std(axis=(0, 1))  # Divide all by 1 mic std
        #     x_spec = x_spec * ((np.sqrt(args.spec_var) / (eps + x_spec_std)))
        #     del x_spec_std
            
        ### Make log values in range [-1,1]
        # x_spec = ((x_spec - x_spec.min()) / (x_spec.max() - x_spec.min())) * 2 - 1
        ### Make log values in range [0,1]
        x_spec = (x_spec - x_spec.min()) / (x_spec.max() - x_spec.min())
        x_spec = np.expand_dims(x_spec, 2)
        
        x = np.concatenate((x_real, x_image, x_spec), axis=-1)
        del x_real, x_image, x_spec
        gc.collect()
        
        if args.std_norm:
            processed_path = args.processed_path + '/std_normalize'
        else:
            processed_path = args.processed_path
        mat_path = '{0}/processed_{1}_{2}.mat'.format(processed_path, inx, n_speak)
        savemat(mat_path, {'x_train':x, 'y_train':y})
    

def main(args):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    preprocess(args)

if __name__ == '__main__':
    main(args)
