from argparse import ArgumentParser

from pytorch_lightning import Trainer, loggers
from torchsummary import summary

from autoencoder import Autoencoder


def main(hparams):
    logger = loggers.TensorBoardLogger(hparams.log_dir,
                                       name=f"bs{hparams.batch_size}_nf{hparams.nfe}",
                                       default_hp_metric=False)

    model = Autoencoder(hparams)

    # print detailed summary with estimated network size
    summary(model, (3, hparams.image_size, hparams.image_size), device="cpu")

    trainer = Trainer(logger=logger,
                      max_epochs=hparams.max_epochs,
                      accelerator='gpu')
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=256, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=500, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size during training")
    parser.add_argument("--nz", type=int, default=2048, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=48, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=48, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate for optimizer")
    parser.add_argument("--data_dir", type=str, default="/home/a.veshkin/data/COCO/CBIR_data/", help="Data dir")

    args = parser.parse_args()
    main(args)
