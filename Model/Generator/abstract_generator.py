import torch.nn as nn
from dataclasses import dataclass, field
from .generator_networks import get_generator_network
from ..Utils.utils_activation import get_activation
from .reconstruction import get_loss_reconstruction


class AbstractGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.reconstruction = get_loss_reconstruction(loss_name=cfg.generator.loss_reconstruction_name, llhd_sigma=cfg.generator.llhd_sigma)
        self.network = get_generator_network(cfg.generator.network_name, cfg.generator.ngf, cfg.trainer.nz, cfg.dataset.nc)
        self.activation = get_activation(cfg.generator.activation_name)

    def forward(self, z):
        z = z.reshape(z.shape[0], self.cfg.trainer.nz, 1, 1,)
        param = self.network(z)
        if self.activation is not None:
            param = self.activation(param)
        return param
    
    def get_loss(self, param, x):
        return self.reconstruction(param, x)
    
    def sample(self, z, return_mean=False):
        z = z.reshape(z.shape[0], self.cfg.trainer.nz, 1, 1,)
        param = self.network(z)
        if self.activation is not None:
            param = self.activation(param)
        return self.reconstruction.sample(param, return_mean=return_mean)



        
    
