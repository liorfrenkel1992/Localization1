import os
import sys
import gc

import numpy as np
import hdf5storage
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class BasicDataset(Dataset):
    def __init__(self, cfg):
        self.eps = sys.float_info.epsilon
        self.K = cfg.K
        self.spec_size = int(self.K / 2)
        deg_delta = cfg.deg_delta
        self.n_class = int(180 / deg_delta + 1)
        self.frame_size = cfg.frame_size
        self.data_path = cfg.data_path
        self.num_microphones = cfg.n_mics
        self.specFixedVar = cfg.specFixedVar
        self.ref_channel = cfg.ref_channel
        self.frames_per_sample = cfg.frames_per_sample
        
        if cfg.input_std_normalize:
            self.data_path = self.data_path + '/std_normalize'
        
        try:
            data_list = os.listdir(self.data_path)[:-1]
            self.ids = [s.split('.')[0] for s in data_list if s.split('.')[1] == 'mat']
        except FileNotFoundError:
            self.ids = []
            
        
        self.training_data = []
        # doas = []
        for img in self.ids:
            img_path = os.path.join(self.data_path, img)
            mat = hdf5storage.loadmat(img_path)
            x, y = mat['x_train'], mat['y_train']
            self.training_data.append((x, y))
            # doas.append(y.flatten())
        
        
        """
        plt.figure()
        plt.hist(doas, bins=40)
        plt.savefig('/mnt/dsi_vol1/users/frenkel2/data/localization/Git/localization1/reports/figures/doa_hist.png')
        plt.close()
        """
        
    def __len__(self):
        return len(self.ids)
    
    def preprocess(self, mat):
        x = mat['x_train']
        y = mat['y_train']
        if x.shape[1] > self.frames_per_sample + 1:
            start_point = np.random.randint(x.shape[1] - self.frames_per_sample + 1)
            x = x[:, start_point:start_point + self.frames_per_sample, :]
            y = y[:, start_point:start_point + self.frames_per_sample, :]
        else:  # Freq dimension is smaller than frames_per_samle
            x_temp = x
            y_temp = y
            x = np.zeros((x_temp.shape[0], self.frames_per_sample, x_temp.shape[2]), dtype=complex)
            y = np.zeros((y_temp.shape[0], self.frames_per_sample, y_temp.shape[2]))
            x[:, :x_temp.shape[1], :] = x_temp
            y[:, :y_temp.shape[1], :] = y_temp
        y = np.where(y > 180, 360 - y, y)
        y[y == -1] = 270
        
        # %% CREATE TRAINING LABELS

        y_sum = np.sum(y, axis=2)
        y_sum_avg = y_sum / self.num_microphones
        VAD_new = y_sum > 270 * (self.num_microphones - 1)
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
        
        y = y[:self.spec_size, :self.frame_size]
        y1 = y.astype(int)

        
        #y = y1.reshape(self.spec_size, self.frame_size, 1)

        for ii in range(self.n_class):
            #y[y1 == ii * 5] = ii + 1
            y[y1 == ii * 5] = ii

        #y[y == 270] = 0
        

        # %% CREATE TRAINING DATA

        x = x + self.eps
        x_spec = np.log(self.eps + np.abs(x[0:self.spec_size, :, self.ref_channel]))
        
        ### Make log values in range [-1,1]
        x_spec = ((x_spec - x_spec.min()) / (x_spec.max() - x_spec.min())) * 2 - 1
        x_spec = np.expand_dims(x_spec, 2)
        x_rtf = x / np.expand_dims(x[:, :, self.ref_channel], 2)
        x_rtf = np.concatenate((x_rtf[:, :, :self.ref_channel], x_rtf[:, :, self.ref_channel + 1:]), axis=-1)
        x_rtf = x_rtf[:self.spec_size, :, :]

        """
        x_rtf_std = np.expand_dims(np.expand_dims(x_rtf.std(axis=(1, 2)), 1), -2)
        x_rtf_std = np.tile(x_rtf_std, (1, spec_size, frame_size, 1))
        x_rtf = x_rtf * ((np.sqrt(1) / (eps + x_rtf_std)))
        """
        x_real = np.real(x_rtf)
        x_image = np.imag(x_rtf)
        
        del x_rtf
        gc.collect()

        """
        ## varaince 1 for every spectrogram
        x_spec_std = np.expand_dims(np.expand_dims(x_spec.std(axis=(1, 2)), 1), -2)
        x_spec_std = np.tile(x_spec_std, (1, spec_size, frame_size, 1))
        x_spec = x_spec * ((np.sqrt(specFixedVar) / (eps + x_spec_std)))
        del x_spec_std
        """

        x = np.concatenate((x_real, x_image, x_spec), axis=-1)
        del x_real, x_image, x_spec
        gc.collect()

        return x, y
        
    def __getitem__(self, i):
        """
        idx = self.ids[i]
        img_path = os.path.join(self.data_path, idx)
        mat = hdf5storage.loadmat(img_path)
        x, y = mat['x_train'], mat['y_train']
        """
        #x, y = self.preprocess(mat)
        x, y = self.training_data[i]
                
        return torch.from_numpy(x.transpose((2, 0, 1))).type(torch.FloatTensor), torch.from_numpy(y).type(torch.LongTensor)
        #return x, y


class LocalizationDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def prepare_data(self):
        BasicDataset(self.cfg)
        
    def setup(self, stage=None):
        dataset = BasicDataset(self.cfg)
        #train_transforms = transforms.Compose([transforms.ToTensor()])
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        self.train, self.val = random_split(dataset, [n_train, n_val])
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, drop_last=True, pin_memory=True)

