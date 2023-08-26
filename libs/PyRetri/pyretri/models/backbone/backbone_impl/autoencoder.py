import torch

from ..backbone_base import BackboneBase
from ...registry import BACKBONES
from .......models.context_encoder.autoencoder import Autoencoder


class AutoencoderBackbone(BackboneBase):
    def __init__(self, model_path):
        super(AutoencoderBackbone, self).__init__()
        self.model = Autoencoder.load_from_checkpoint(model_path)
        self.model.eval()

    def forward(self, x):
        return self.model.encoder(x)


@BACKBONES.register
def autoencoder(model_path):
    model = AutoencoderBackbone(model_path)
    return model
