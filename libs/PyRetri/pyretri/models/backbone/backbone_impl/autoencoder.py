import torchvision.transforms

from ..backbone_base import BackboneBase
from ...registry import BACKBONES

import pytorch_lightning as pl
import torch.nn as nn
import torch


CHECKPOINT_PATH = '/home/artem/dev/CBIR-v2/models/context_encoder/checkpoints/bs64_nf8_nz2048_convs_in_block_1.ckpt'


class Autoencoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.init_model(self.hparams)

    @staticmethod
    def encoder_block(in_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, n_additional_convs=0):
        blocks = []
        for _ in range(n_additional_convs):
            blocks += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(True),
            ]

        blocks += [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        ]

        return nn.Sequential(*blocks)

    @staticmethod
    def decoder_block(in_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, n_additional_convs=0, last_relu=True):
        block = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels)
        ]

        for _ in range(n_additional_convs):
            block += [
                nn.LeakyReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ]

        if last_relu:
            block.append(nn.LeakyReLU(True))

        return nn.Sequential(*block)

    def init_model(self, hparams):
        nf = hparams.nf
        nz  = hparams.nz
        n_additional_convs = hparams.n_additional_convs
        self.encoder = nn.Sequential(
            # input (nc) x 256 x 256
            self.encoder_block(3, nf, 4, 2, 1),
            # input (nf) x 128 x 128
            self.encoder_block(nf, nf * 2, 4, 2, 1, n_additional_convs),
            # input (nf*2) x 64 x 64
            self.encoder_block(nf * 2, nf * 4, 4, 2, 1, n_additional_convs),
            # input (nf*4) x 32 x 32
            self.encoder_block(nf * 4, nf * 8, 4, 2, 1, n_additional_convs),
            # input (nf*8) x 16 x 16
            self.encoder_block(nf * 8, nf * 16, 4, 2, 1, n_additional_convs),
            # input (nf*16) x 8 x 8
            self.encoder_block(nf * 16, nf * 32, 4, 2, 1, n_additional_convs),
            # input (nf*32) x 4 x 4
            self.encoder_block(nf * 32, nz, 4, 1, 0),
            # output (nz) x 1 x 1
        )

        self.decoder = nn.Sequential(
            # input (nz) x 1 x 1
            self.decoder_block(nz, nf * 32, 4, 1, 0),
            # input (nf*32) x 4 x 4
            self.decoder_block(nf * 32, nf * 16, 4, 2, 1, n_additional_convs),
            # input (nf*16) x 8 x 8
            self.decoder_block(nf * 16, nf * 8, 4, 2, 1, n_additional_convs),
            # input (nf*8) x 16 x 16
            self.decoder_block(nf * 8, nf * 4, 4, 2, 1, n_additional_convs),
            # input (nf*4) x 32 x 32
            self.decoder_block(nf * 4, nf * 2, 4, 2, 1, n_additional_convs),
            # input (nf*2) x 64 x 64
            self.decoder_block(nf * 2, nf, 4, 2, 1, n_additional_convs),
            # input (nf) x 128 x 128
            self.decoder_block(nf, 3, 4, 2, 1, last_relu=False),
            nn.Tanh()
            # output (nc) x 256 x 256
        )


class AutoencoderBackbone(BackboneBase):
    def __init__(self):
        super(AutoencoderBackbone, self).__init__()
        self.model = Autoencoder.load_from_checkpoint(CHECKPOINT_PATH)
        self.model.eval()

    def forward(self, x):
        self.model.eval()
        return self.model.encoder(x)


@BACKBONES.register
def autoencoder():
    model = AutoencoderBackbone()
    return model
