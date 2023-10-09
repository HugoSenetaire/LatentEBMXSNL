import torch.nn as nn
import torch


class GaussianPrior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mu = nn.Parameter(torch.zeros((self.cfg.trainer.nz), device=self.cfg.trainer.device), requires_grad=False)
        self.std = nn.Parameter(torch.ones((self.cfg.trainer.nz), device=self.cfg.trainer.device), requires_grad=False)
              
    def log_prob(self, z):
        """By default return the gaussian 0,1"""
        return torch.distributions.Normal(self.mu, self.std).log_prob(z).reshape(z.shape[0], self.cfg.trainer.nz).sum(1)
    
    
    def sample(self, n):
        return torch.distributions.Normal(self.mu, self.std).sample((n,))

