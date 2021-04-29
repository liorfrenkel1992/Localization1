import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn as nn

import pytorch_lightning as pl


class UNet(pl.LightningModule):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.in_channels = cfg.in_channels
        self.net_fact = cfg.net_fact
        self.out_channels = cfg.n_classes
        self.lr = cfg.lr
        self.beta2 = cfg.beta2
        self.beta1 = cfg.beta1
        self.eps = cfg.eps
        self.momentum = cfg.momentum
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=270)
        self.optim = cfg.optim
        
        self.save_hyperparameters()

        class DoubleConv(nn.Module):
            def __init__(self, in_channels, out_channels, dropout_fact):
                super(DoubleConv, self).__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.dropout_fact = dropout_fact

                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout_fact),
                    # nn.BatchNorm2d(mid_channels),
                    # nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(0.1)
                    # nn.BatchNorm2d(out_channels),
                    # nn.ReLU(inplace=True)
                )

            def forward(self, x):
                return self.double_conv(x)

        class Down(nn.Module):
            def __init__(self, in_channels, out_channels, dropout_fact):
                super(Down, self).__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels

                self.maxpool_conv = nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(in_channels, out_channels, dropout_fact)
                )

            def forward(self, x):
                return self.maxpool_conv(x)

            """
                    w = torch.empty(3, 3)
                    w = nn.init.kaiming_normal_(w)

                    c1 = nn.Conv2d(in_channels, 16*net_fact, (3, 3))
                    c1 = nn.ELU(c1)
                    with torch.no_grad():
                        c1.weight = nn.Parameter(w)
                    """

        class Up(nn.Module):
            """Upscaling then double conv"""

            def __init__(self, in_channels, out_channels, dropout_fact):
                super(Up, self).__init__()

                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels, dropout_fact)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # input is CHW
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)

        class OutConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(OutConv, self).__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                #self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                return self.conv(x)
                #return self.softmax(out)

        self.inc = DoubleConv(self.in_channels, 16 * self.net_fact, 0.1)
        self.down1 = Down(16 * self.net_fact, 32 * self.net_fact, 0.1)
        self.down2 = Down(32 * self.net_fact, 64 * self.net_fact, 0.2)
        self.down3 = Down(64 * self.net_fact, 128 * self.net_fact, 0.2)
        self.down4 = Down(128 * self.net_fact, 256 * self.net_fact, 0.3)
        self.up1 = Up(256 * self.net_fact, 128 * self.net_fact, 0.2)
        self.up2 = Up(128 * self.net_fact, 64 * self.net_fact, 0.2)
        self.up3 = Up(64 * self.net_fact, 32 * self.net_fact, 0.1)
        self.up4 = Up(32 * self.net_fact, 16 * self.net_fact, 0.1)
        self.outputs = OutConv(16 * self.net_fact, self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outputs(x)

        return logits

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        #tensorboard_logs = {'train_loss': loss}
        #return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        #return {'val_loss': loss}
        
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    

    def configure_optimizers(self):
        if self.optim == 'adam':
            return optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=self.eps)  # eps=1e-07
        elif self.optim == 'sgd':
            return optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        