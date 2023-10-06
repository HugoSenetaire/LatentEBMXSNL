import torch.nn as nn
import torch


class GaussianPrior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mu = nn.Parameter(torch.tensor(0.,), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(1.,), requires_grad=False)
       
    def log_prob(self, z):
        """By default return the gaussian 0,1"""
        return torch.distributions.Normal(self.mu, self.std).log_prob(z)
    
    def sample(self, n):
        return torch.randn((n, self.cfg.trainer.nz, 1, 1))

