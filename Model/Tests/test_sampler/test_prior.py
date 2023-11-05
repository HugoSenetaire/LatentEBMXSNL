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

from Model.Sampler import get_prior_sampler, get_prior_sampler
from Model.Prior import get_prior



class DummyEnergy(nn.Module):
    def __init__(self, norm_control = 10) -> None:
        super().__init__()
        self.norm_control = norm_control

    def forward(self, z):
        energy = -(z.norm(dim=-1, keepdim=True)) * self.norm_control
        # print(energy.mean()/self.norm_control)
        return energy




def test_prior():
    cfg = get_config(config_name="conf")
    cfg.trainer.nz = 2
    for prior_name, num_samples in zip(["langevin", "nuts", "mala"], [100, 100, 100]):
        cfg.sampler_prior.sampler_name = prior_name
        cfg.sampler_prior.num_samples = num_samples
        cfg.sampler_prior.thinning = 1
        cfg.sampler_prior.warmup_steps = 200
        cfg.sampler_prior.step_size = 0.3
        prior_sampler = get_prior_sampler(cfg.sampler_prior)
        base_dist = get_prior(cfg.trainer.nz, cfg.prior)
        z_init_small = base_dist.sample(10)
        z_init_large = base_dist.sample(10)
        energy_small = DummyEnergy(norm_control=1)
        energy_large = DummyEnergy(norm_control=10)

        z_samples_small = prior_sampler(z = z_init_small, energy= energy_small, base_dist= base_dist,)
        z_samples_large = prior_sampler(z = z_init_large, energy= energy_large, base_dist= base_dist,)

        assert z_samples_large.shape == (1000, 2), "The shape should be (10000, 2)"
        assert z_samples_large.norm(dim=-1, keepdim=True).mean()>z_samples_small.norm(dim=-1,keepdim=True).mean(), "Larger samples should have a larger norm for {}".format(prior_name)


if __name__ == "__main__":
    test_prior()