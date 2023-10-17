import torch
import torch.nn as nn


class MixtureOfGaussian(nn.Module):
    def __init__(self, nz, cfg):
        super().__init__()
        self.nz = nz
        self.cfg = cfg
        self.log_mix = torch.nn.Parameter(torch.randn((self.cfg.nb_mixture))/self.cfg.nb_mixture)
        self.mu = torch.nn.Parameter(torch.randn((self.cfg.nb_mixture, self.nz)))
        self.log_var = torch.nn.Parameter(torch.randn((self.cfg.nb_mixture, self.nz)))

    def log_prob(self, z):
        z = z.flatten(1)
        component_prob = (self.log_mix - torch.logsumexp(self.log_mix, dim=0)).exp()
        self.mix_dist = torch.distributions.categorical.Categorical(component_prob)
        self.comp_dist = torch.distributions.Independent(torch.distributions.normal.Normal(self.mu, torch.exp(self.log_var)),1)
        self.gmm = torch.distributions.mixture_same_family.MixtureSameFamily(self.mix_dist, self.comp_dist)
        return self.gmm.log_prob(z).reshape(z.shape[0])
    
    def sample(self, n):
        component_prob = (self.log_mix - torch.logsumexp(self.log_mix, dim=0)).exp()
        self.mix_dist = torch.distributions.categorical.Categorical(component_prob)
        self.comp_dist = torch.distributions.Independent(torch.distributions.normal.Normal(self.mu, torch.exp(self.log_var)),1)
        self.gmm = torch.distributions.mixture_same_family.MixtureSameFamily(self.mix_dist, self.comp_dist)
        return self.gmm.sample((n,))
