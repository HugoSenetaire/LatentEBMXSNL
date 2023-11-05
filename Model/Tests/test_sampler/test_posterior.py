import sys
import os
import torch
import math
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
current_path.split('L')[0]
sys.path.append(current_path.split('Model')[0])
from Model.Tests.utils_test import get_config
import torch.nn as nn

from Model.Sampler import get_prior_sampler, get_posterior_sampler
from Model.Prior import get_prior


class DummyGenerator(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
       
    def forward(self, z):
        expand_z_norm = z.norm(dim=-1, keepdim=True).expand(-1, self.size)
        return (torch.tanh(expand_z_norm-5) + 1) / 2
        # return torch.where(expand_z_norm > 1.0, torch.zeros_like(z), torch.ones_like(z))
        
    def get_loss(self, param, x, dic_params=None):
        assert param.shape == x.shape
        return -torch.distributions.Bernoulli(param,).log_prob(x).sum(dim=-1)

class DummyEnergy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, z):
        return torch.zeros(z.shape[0])




def test_posterior():
    cfg = get_config(config_name="conf")
    cfg.trainer.nz = 2
    for posterior_name, num_samples in zip(["langevin", "nuts", "mala"], [100, 100, 100]):
        cfg.sampler_posterior.sampler_name = posterior_name
        cfg.sampler_posterior.num_samples = num_samples
        cfg.sampler_posterior.thinning = 1
        cfg.sampler_posterior.warmup_steps = 100
        cfg.sampler_posterior.step_size = 0.3
        posterior_sampler = get_posterior_sampler(cfg.sampler_posterior)
        base_dist = get_prior(cfg.trainer.nz, cfg.prior)
        z_init_large= base_dist.sample(10)
        z_init_small = base_dist.sample(10)
        # base_dist = DummyBaseDist()
        energy = DummyEnergy()
        generator = DummyGenerator(cfg.trainer.nz)

        samples_big = torch.ones(10, cfg.trainer.nz)
        z_samples_big = posterior_sampler(z = z_init_large,x = samples_big, energy= energy, base_dist= base_dist,generator = generator)
        samples_small = torch.zeros(10, cfg.trainer.nz)
        z_samples_small = posterior_sampler(z = z_init_small,x = samples_small, energy= energy, base_dist= base_dist,generator = generator)

        assert z_samples_big.shape == (1000, 2), "The shape should be (10000, 2)"
        print(z_samples_big.norm(dim=-1, keepdim=True).mean())
        print(z_samples_small.norm(dim=-1, keepdim=True).mean())
        assert z_samples_big.norm(dim=-1, keepdim=True).mean()>z_samples_small.norm(dim=-1,keepdim=True).mean(), "Larger samples should have a larger norm for {}".format(posterior_name)


if __name__ == "__main__":
    test_posterior()