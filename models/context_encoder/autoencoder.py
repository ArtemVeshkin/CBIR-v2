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

        self.init_model(hparams)

        self.val_batch_masks = []
        self.test_batch_masks = []

    def init_model(self, hparams):
        self.encoder = nn.Sequential(
            # input 3 x 256 x 256
            nn.Conv2d(3, hparams.nfe, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe),
            nn.LeakyReLU(True),
            # input (nfe) x 128 x 128
            nn.Conv2d(hparams.nfe, hparams.nfe * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 2),
            nn.LeakyReLU(True),
            # input (nfe*2) x 64 x 64
            nn.Conv2d(hparams.nfe * 2, hparams.nfe * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 4),
            nn.LeakyReLU(True),
            # input (nfe*4) x 32 x 32
            nn.Conv2d(hparams.nfe * 4, hparams.nfe * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 8),
            nn.LeakyReLU(True),
            # input (nfe*8) x 16 x 16
            nn.Conv2d(hparams.nfe * 8, hparams.nfe * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 16),
            nn.LeakyReLU(True),
            # input (nfe*16) x 8 x 8
            nn.Conv2d(hparams.nfe * 16, hparams.nfe * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 32),
            nn.LeakyReLU(True),
            # input (nfe*32) x 4 x 4
            nn.Conv2d(hparams.nfe * 32, hparams.nfe * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfe * 64),
            nn.LeakyReLU(True),
            # input (nfe*64) x 2 x 2
            nn.Conv2d(hparams.nfe * 64, hparams.nz, 2, 1, 0, bias=False),
            nn.BatchNorm2d(hparams.nz),
            nn.LeakyReLU(True)
            # output (nz) x 1 x 1
        )

        self.decoder = nn.Sequential(
            # input (nz) x 1 x 1
            nn.ConvTranspose2d(hparams.nz, hparams.nfd * 64, 2, 1, 0, bias=False),
            nn.BatchNorm2d(hparams.nfd * 64),
            nn.ReLU(True),
            # input (nfd*64) x 2 x 2
            nn.ConvTranspose2d(hparams.nfd * 64, hparams.nfd * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 32),
            nn.ReLU(True),
            # input (nfd*32) x 4 x 4
            nn.ConvTranspose2d(hparams.nfd * 32, hparams.nfd * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 16),
            nn.ReLU(True),
            # input (nfd*16) x 8 x 8
            nn.ConvTranspose2d(hparams.nfd * 16, hparams.nfd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 8),
            nn.ReLU(True),
            # input (nfd*8) x 16 x 16
            nn.ConvTranspose2d(hparams.nfd * 8, hparams.nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 4),
            nn.ReLU(True),
            # input (nfd*4) x 32 x 32
            nn.ConvTranspose2d(hparams.nfd * 4, hparams.nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 2),
            nn.ReLU(True),
            # input (nfd*2) x 64 x 64
            nn.ConvTranspose2d(hparams.nfd * 2, hparams.nfd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd),
            nn.ReLU(True),
            # input (nfd) x 128 x 128
            nn.ConvTranspose2d(hparams.nfd, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # output 3 x 256 x 256
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

        self.train_dataset = FolderDataset(data_dir='/home/artem/data/COCO/CBIR_data/train',
                                           load_fn=np.load,
                                           transform_fn=transform)
        self.val_dataset = FolderDataset(data_dir='/home/artem/data/COCO/CBIR_data/val',
                                         load_fn=np.load,
                                         transform_fn=transform)

        self.test_dataset = FolderDataset(data_dir='/home/artem/data/COCO/CBIR_data/test',
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

        masked_output = [denormalization(i) * mask + denormalization(out) * (1. - mask)
                  for i, out, mask in zip(x[:n], output[:n], batch_mask[:n])]

        raw_output = [denormalization(i) for i in output[:n]]

        # make grids and save to logger
        grid_input = vutils.make_grid(input, nrow=n)
        grid_masked_input = vutils.make_grid(masked_input, nrow=n)
        grid_masked_output = vutils.make_grid(masked_output, nrow=n)
        grid_raw_output = vutils.make_grid(raw_output, nrow=n)
        grid = torch.cat((grid_input, grid_masked_input, grid_masked_output, grid_raw_output), 1)
        self.logger.experiment.add_image(name, grid, self.global_step)

    def generate_batch_mask(self, batch):
        bs, _, w, h = batch.shape
        fake_batch = np.ones((bs, w, h, 3), dtype=np.float32)
        cutout = iaa.Sequential([iaa.Cutout(size=0.35, nb_iterations=(2, 5),
                                            fill_mode='constant', cval=0.)])
        batch_mask = cutout(images=fake_batch)
        batch_mask = batch_mask.transpose(0, 3, 1, 2)
        batch_mask = torch.Tensor(batch_mask).to(self.device)

        return batch_mask

    def training_step(self, batch, batch_idx):
        batch_mask = self.generate_batch_mask(batch)
        masked_batch = batch * batch_mask
        output = self(masked_batch)
        loss = F.mse_loss(output * (1. - batch_mask), batch * (1. - batch_mask)) * batch_mask.mean()

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
        output = self(masked_batch)
        loss = (F.mse_loss(output * (1. - batch_mask), batch * (1. - batch_mask)) * batch_mask.mean()).detach().cpu()

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
        output = self(masked_batch)
        loss = (F.mse_loss(output * (1. - batch_mask), batch * (1. - batch_mask)) * batch_mask.mean()).detach().cpu()

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(batch, output, batch_mask, "test_input_output")

        self.test_loss += [loss]
        batch_mask = batch_mask.detach().cpu()

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_loss).mean()
        self.logger.experiment.add_scalars('loss',
                                           {'test': avg_loss}, self.global_step)