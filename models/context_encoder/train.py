from argparse import ArgumentParser

from pytorch_lightning import Trainer, loggers
from torchsummary import summary

from autoencoder import Autoencoder

import os


def main(hparams):
    exp_name = (f"bs{hparams.batch_size}_"
                f"nf{hparams.nf}_"
                f"nz{hparams.nz}_"
                f"convs_in_block_{hparams.n_additional_convs + 1}_"
                f"masked_mask_0.15_1-mask")
    logger = loggers.TensorBoardLogger(hparams.log_dir,
                                       name=exp_name,
                                       default_hp_metric=False)

    model = Autoencoder(hparams)
    # model = Autoencoder.load_from_checkpoint('/home/artem/dev/CBIR-v2/models/context_encoder/checkpoints/bs64_nf8_nz2048_convs_in_block_1.ckpt')

    # print detailed summary with estimated network size
    summary(model, (4, hparams.image_size, hparams.image_size), device="cpu")

    trainer = Trainer(logger=logger,
                      max_epochs=hparams.max_epochs,
                      accelerator='gpu')
    trainer.fit(model)
    trainer.test(model)
    trainer.save_checkpoint(f"checkpoints/{exp_name}.ckpt")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=256, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=200, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size during training")
    parser.add_argument("--nz", type=int, default=2048, help="Size of latent vector z")
    parser.add_argument("--nf", type=int, default=16, help="Size of feature maps in encoder/decoder")
    parser.add_argument("--n_additional_convs", type=int, default=0, help="N additional conv layers in block")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate for optimizer")
    parser.add_argument("--data_dir", type=str, default=f"{os.getenv('HOME')}/data/COCO/CBIR_data/", help="Data dir")

    args = parser.parse_args()
    main(args)
