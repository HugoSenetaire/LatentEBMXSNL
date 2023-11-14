import torch.nn as nn
import torch
import math

class StudentProposal(nn.Module):
    def __init__(self, nz, cfg,):
        super().__init__()
        self.nz = nz
        self.cfg = cfg
        self.df = nn.Parameter(torch.full((1,), fill_value=cfg.df,), requires_grad=False)
        self.mu = nn.Parameter(torch.full((self.nz,), fill_value=cfg.mu,), requires_grad=False)
        self.log_var = nn.Parameter(torch.full((self.nz,), fill_value=2*math.log(cfg.sigma)), requires_grad=False)

    def log_prob(self, z):
        return torch.distributions.StudentT(self.df, self.mu, (0.5 * self.log_var).exp()).log_prob(z).reshape(z.shape[0], self.nz).sum(1)

    def sample(self, n):
        return torch.distributions.StudentT(self.df, self.mu, (0.5 * self.log_var).exp()).sample((n,))
    