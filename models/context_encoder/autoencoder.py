import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import FolderDataset
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

# normalization constants
MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


class Autoencoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.init_model(self.hparams)

        self.val_batch_masks = []
        self.test_batch_masks = []

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
        print(self.encoder)

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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def prepare_data(self):

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.hparams.image_size),
                transforms.CenterCrop(self.hparams.image_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN.tolist(), STD.tolist()),
            ]
        )

        self.train_dataset = FolderDataset(data_dir=f'{self.hparams.data_dir}train',
                                           load_fn=np.load,
                                           transform_fn=transform)
        self.val_dataset = FolderDataset(data_dir=f'{self.hparams.data_dir}val',
                                         load_fn=np.load,
                                         transform_fn=transform)

        self.test_dataset = FolderDataset(data_dir=f'{self.hparams.data_dir}/test',
                                          load_fn=np.load,
                                          transform_fn=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def configure_optimizers(self):
        return Adam(self.parameters(),
                    lr=self.hparams.lr,
                    betas=(0.9, 0.999))

    def save_images(self, x, output, batch_mask, name, n=8):
        """
        Saves a plot of n images from input and output batch
        """
        n = min(n, self.hparams.batch_size)

        # denormalize images
        denormalization = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
        input = [denormalization(i) for i in x[:n]]
        masked_input = [(denormalization(i) + (1 - mask)).clamp(0., 1.)
                        for i, mask in zip(x[:n], batch_mask[:n])]
        raw_output = [denormalization(i) for i in output[:n]]

        # make grids and save to logger
        grid_input = vutils.make_grid(input, nrow=n)
        grid_masked_input = vutils.make_grid(masked_input, nrow=n)
        grid_raw_output = vutils.make_grid(raw_output, nrow=n)
        grid = torch.cat((grid_input, grid_masked_input, grid_raw_output), 1)
        self.logger.experiment.add_image(name, grid, self.global_step)

    def generate_batch_mask(self, batch):
        bs, _, w, h = batch.shape
        fake_batch = np.ones((bs, w, h, 3), dtype=np.float32)
        cutout = iaa.Sequential([iaa.Cutout(size=0.15, nb_iterations=(2, 5),
                                            fill_mode='constant', cval=0.)])
        batch_mask = cutout(images=fake_batch)
        batch_mask = batch_mask.transpose(0, 3, 1, 2)
        batch_mask = torch.Tensor(batch_mask).to(self.device)

        return batch_mask

    def training_step(self, batch, batch_idx):
        batch_mask = self.generate_batch_mask(batch)
        masked_batch = batch * batch_mask
        # output = self(masked_batch)
        output = self(torch.cat((masked_batch,
                                 1 - batch_mask[:, 0:1, :, :]), 1))
        loss = F.mse_loss(output, batch)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(batch, output, batch_mask, "train_input_output")

        self.logger.experiment.add_scalars('loss',
                                           {'train': loss}, self.global_step)

        masked_batch.detach().cpu()
        return {
            'loss': loss
        }

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_loss = []

    def validation_step(self, batch, batch_idx):
        if len(self.val_batch_masks) < batch_idx + 1:
            self.val_batch_masks.append(self.generate_batch_mask(batch).detach().cpu())
        batch_mask = self.val_batch_masks[batch_idx].to(self.device)

        masked_batch = batch * batch_mask
        # output = self(masked_batch)
        output = self(torch.cat((masked_batch,
                                 1 - batch_mask[:, 0:1, :, :]), 1))
        loss = (F.mse_loss(output, batch)).detach().cpu()

        # save input and output images at beginning of epoch
        if batch_idx == 1:
            self.save_images(batch, output, batch_mask, "val_input_output")

        self.val_loss += [loss]
        batch_mask = batch_mask.detach().cpu()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        self.logger.experiment.add_scalars('loss',
                                           {'val': avg_loss}, self.global_step)

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_loss = []

    def test_step(self, batch, batch_idx):
        if len(self.test_batch_masks) < batch_idx + 1:
            self.test_batch_masks.append(self.generate_batch_mask(batch).detach().cpu())
        batch_mask = self.test_batch_masks[batch_idx].to(self.device)
        masked_batch = batch * batch_mask
        # output = self(masked_batch)
        output = self(torch.cat((masked_batch,
                                 1 - batch_mask[:, 0:1, :, :]), 1))
        loss = (F.mse_loss(output, batch)).detach().cpu()

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(batch, output, batch_mask, "test_input_output")

        self.test_loss += [loss]
        batch_mask = batch_mask.detach().cpu()

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_loss).mean()
        self.logger.experiment.add_scalars('loss',
                                           {'test': avg_loss}, self.global_step)