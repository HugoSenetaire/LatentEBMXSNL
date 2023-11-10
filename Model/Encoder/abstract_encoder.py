import torch.nn as nn
from dataclasses import dataclass
from .encoder_networks import get_encoder_network
from .latent_distribution import get_latent_distribution
from ..Utils.utils_activation import get_activation
import numpy as np 
import tqdm
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
        self.nz = nz
        self.latent_distribution = get_latent_distribution(cfg.encoder.latent_distribution_name, cfg, )
        self.lambda_nz = self.latent_distribution.lambda_nz
        self.network = get_encoder_network(cfg.encoder.network_name, cfg.encoder.nef, nz, nc, lambda_nz=self.lambda_nz)
        self.activation = get_activation(cfg.encoder.activation_name)
        self.mean_init = torch.nn.parameter.Parameter(torch.zeros(self.lambda_nz(nz)), requires_grad=False)

    def forward(self, x,):

        param = self.network(x)-self.mean_init.unsqueeze(0) 
        # param -= self.mean_init.unsqueeze(0)
        if self.activation is not None:
            param = self.activation(param)
        return param

    def init_network(self, train_loader, opt_encoder=None):
        """
        Initialize the network with the mean of the prior
        """
        if self.cfg.encoder.init_network_type is None :
            pass
        elif self.cfg.encoder.init_network_type == 'mean':
            with torch.no_grad():
                running_mean = torch.zeros(self.lambda_nz(self.nz), device=self.cfg.trainer.device)
                running_variance = torch.ones(self.lambda_nz(self.nz), device=self.cfg.trainer.device)
                running_count = 0
                ranger = tqdm.tqdm(enumerate(train_loader), desc="Init network to mean")
                for batch_idx, (data, target) in ranger:
                    data = data.to(self.cfg.trainer.device)
                    param = self.network(data).detach().clone()
                    current_count = data.shape[0]
                    running_mean = running_mean*(running_count/(running_count+current_count)) + param.sum(dim=0)/(running_count+current_count)
                    running_variance = running_variance*(running_count/(running_count+current_count)) + ((param**2).sum(dim=0)/(running_count+current_count)) - running_mean**2
                    running_count += current_count
                self.mean_init.data = running_mean
        elif self.cfg.encoder.init_network_type == 'mean_variance':
            with torch.no_grad():
                running_mean = torch.zeros(self.lambda_nz(self.nz), device=self.cfg.trainer.device)
                running_variance = torch.ones(self.lambda_nz(self.nz), device=self.cfg.trainer.device)
                running_count = 0
                ranger = tqdm.tqdm(enumerate(train_loader), desc="Init network to mean")
                for batch_idx, (data, target) in ranger:
                    data = data.to(self.cfg.trainer.device)
                    param = self.network(data).detach().clone()
                    current_count = data.shape[0]
                    running_mean = running_mean*(running_count/(running_count+current_count)) + param.sum(dim=0)/(running_count+current_count)
                    running_variance = running_variance*(running_count/(running_count+current_count)) + ((param**2).sum(dim=0)/(running_count+current_count)) - running_mean**2
                    running_count += current_count
                self.mean_init.data = running_mean
            for batch in train_loader:
                data = batch[0].to(self.cfg.trainer.device)
                param = self.network(data).detach().clone()
                param -= self.mean_init.unsqueeze(0)
                self.latent_distribution.init_network(param)

        else :
            raise NotImplementedError(f"init_network_type {self.cfg.encoder.init_network_type} is not implemented")
        