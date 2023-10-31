import torch.nn as nn
import torch


class UniformPrior(nn.Module):
    def __init__(self, nz, cfg):
        super().__init__()
        self.cfg = cfg
        self.nz = nz
        current_min = torch.full((self.nz,), self.cfg.min,)
        current_max = torch.full((self.nz,), self.cfg.max,)
        self.min = nn.Parameter(current_min, requires_grad=False)
        self.max = nn.Parameter(current_max, requires_grad=False)

    def log_prob(self, z):
        """By default return the gaussian 0,1"""
        assert torch.all(z >= self.min) and torch.all(z <= self.max), "z should be in the range of [min, max]"
        return torch.distributions.Uniform(self.min, self.max).log_prob(z).reshape(z.shape[0], self.nz).sum(1)
    
    def sample(self, n):
        return torch.distributions.Uniform(self.min, self.max).sample((n,))
    

    
    
