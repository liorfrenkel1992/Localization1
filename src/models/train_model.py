import os
import sys
import gc

import torch
import numpy as np
from torch import nn, optim

from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TestTubeLogger

import hdf5storage

from omegaconf import DictConfig, OmegaConf
import hydra

from unet import UNet
from dataset_module import LocalizationDataModule

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"


@hydra.main(config_name='train_cfg')
def main(cfg: DictConfig) -> None:
    
    dm = LocalizationDataModule(cfg)
    model = UNet(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename='bs_{0}_lr_{1}'.format(cfg.batch_size, cfg.lr),
        verbose=cfg.verbose
    )

    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=5,
        verbose=cfg.verbose,
    )

    trainer = Trainer(
        gpus=[cfg.gpu_device],
        callbacks=[checkpoint_callback, stop_callback],
        accelerator="ddp", precision=16
        #max_epochs=cfg.epochs
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint(os.path.join(cfg.checkpoint_dir, 'unet_realRTF_imgRTF_spect_16Khz_4micsArray_bs_{0}_lr_{1}_Lior.ckpt'.format(cfg.batch_size, cfg.lr)))
    

if __name__ == "__main__":
    main()