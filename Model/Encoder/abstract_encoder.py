import torch.nn as nn
from dataclasses import dataclass
from .encoder_networks import get_encoder_network
from .latent_distribution import get_latent_distribution
from ..Utils.utils_activation import get_activation

import torch


class AbstractEncoder(nn.Module):
    """
    Class that defines the encoder. Just choose the network and the activation which should be None now.
    """

    def __init__(self, cfg, nz, nc, reverse = False):
        super().__init__()
        self.cfg = cfg
        self.reverse = reverse
        self.count =0
        self.latent_distribution = get_latent_distribution(cfg.encoder.latent_distribution_name, cfg, )
        self.lambda_nz = self.latent_distribution.lambda_nz
        self.network = get_encoder_network(cfg.encoder.network_name, cfg.encoder.nef, nz, nc, lambda_nz=self.lambda_nz)
        self.activation = get_activation(cfg.encoder.activation_name)

    def forward(self, x,):
        param = self.network(x)
        if self.activation is not None:
            param = self.activation(param)
        return param
