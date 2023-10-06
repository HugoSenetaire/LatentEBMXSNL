import torch.nn as nn
from dataclasses import dataclass
from .encoder_networks import get_encoder_network
from ..Utils.utils_activation import get_activation
import torch


class AbstractEncoder(nn.Module):
    """
    Class that defines the encoder. Just choose the network and the activation which should be None now.
    """

    def __init__(self, cfg, nz, nc):
        super().__init__()
        self.cfg = cfg
        self.network = get_encoder_network(cfg.encoder.network_name, cfg.encoder.nef, nz, nc)
        self.activation = get_activation(cfg.encoder.activation_name)

    def forward(self, x):
        param = self.network(x)
        if self.activation is not None:
            param = self.activation(param)
        return param
    
   
    def sample(self, x = None, param = None):
        """
        Sample from the approximate posterior given a x, or directly form the param
        """
        if param is None :
            param = self.network(x)
            if self.activation is not None:
                param = self.activation(param)
        else :
            if x is None :
                raise AttributeError("x is None but param is not None")
        mu, log_var = param.flatten(1).chunk(2, dim=1)
        x = torch.normal(mu, torch.exp(0.5*log_var))
        return x


        
    
