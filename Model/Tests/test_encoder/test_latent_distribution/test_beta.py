import sys
import os
import torch
import math
current_path = os.path.dirname(os.path.realpath(__file__))
current_path.split('L')[0]
sys.path.append(current_path.split('Model')[0])
from Model.Tests.utils_test import get_config

from Model.Encoder.latent_distribution import get_latent_distribution
from Model.Prior import get_prior

from Model.Encoder.latent_distribution.beta import BetaPosterior
from Model.Prior.uniform_prior import UniformPrior



def test_beta():
    cfg = get_config(config_name="conf_uniform")
    cfg.prior.prior_name = 'uniform'
    cfg.encoder.latent_distribution_name = 'beta'
    cfg.prior.min = 0.0
    cfg.prior.max = 1.0
    beta_posterior = get_latent_distribution(cfg.encoder.latent_distribution_name, cfg)
    uniform_prior = get_prior(cfg.trainer.nz, cfg.prior)
    assert isinstance(beta_posterior, BetaPosterior), "The beta posterior should be a betaPosterior"
    assert isinstance(uniform_prior, UniformPrior), "The uniform prior should be a UniformPrior"

    param = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0]], dtype=torch.float32).log()
    dic_params, dic_params_feedback = beta_posterior.get_params(param)

    #Verify the obtention of parameters
    print(dic_params_feedback["alpha_avg"])
    assert dic_params_feedback["alpha_avg"].shape == (3,), "The min should be a vector of size 2"
    assert torch.allclose(dic_params_feedback["alpha_avg"], param[:,:2].exp().mean(1)), "The min should be -1"
    assert torch.allclose(dic_params_feedback["beta_avg"], param[:,2:].exp().mean(1)), "The min should be -1"
    assert torch.allclose(dic_params["alpha"], param.exp().chunk(2,1)[0]), "The min should be -1"
    assert torch.allclose(dic_params["beta"],  param.exp().chunk(2,1)[0]), "The max should be 1"

    #Verify log_prob
    sample = torch.tensor([[0.5, 0.5],[0.5, 0.5], [0.5, 0.5]])
    dist1 = torch.distributions.Beta(1,1)
    dist2 = torch.distributions.Beta(0.5,0.5)
    dist3 = torch.distributions.Beta(2,2)
    target_prob = [dist1.log_prob(sample[0,0])+ dist1.log_prob(sample[0,1]), dist2.log_prob(sample[1,0])+ dist2.log_prob(sample[1,1]), dist3.log_prob(sample[2,0])+ dist3.log_prob(sample[2,1])]

    assert torch.allclose(beta_posterior.log_prob(param, sample), torch.stack(target_prob)), "The log prob is weird"
    sample_expanded = sample.unsqueeze(0).expand(2, -1, -1).flatten(0, 1)
    assert sample_expanded.shape == (6, 2), "The shape should be (6, 2)"
    assert beta_posterior.log_prob(param, sample_expanded).shape == (2,3), "The shape should be (2,3)"
    assert torch.allclose(beta_posterior.log_prob(param, sample_expanded).flatten(), torch.stack(target_prob*2)), "The log prob is weird"

    #Verify sample
    samples_posterior = beta_posterior.r_sample(param, 1000, dic_params=dic_params)
    assert torch.all(samples_posterior >= cfg.prior.min) and torch.all(samples_posterior <= cfg.prior.max), "The samples should be in the range of [min, max]"
    assert samples_posterior.shape == (1000, 3, 2), "The samples should be of size (1000, 2)"

    # Test for the KL
    # Backward KL :
    kl_empirical = beta_posterior.calculate_kl(uniform_prior, param, samples_posterior.flatten(0,1), dic_params=dic_params, empirical_kl=True)
    
    assert(kl_empirical.shape == (3,)), "The kl should be a vector of size 2"
    assert kl_empirical[0]< kl_empirical[1], "kl uniform should be the smallest"
    assert kl_empirical[0]< kl_empirical[2], "kl uniform should be the smallest"


    # Test entropy :
    entropy_emp1 = beta_posterior.calculate_entropy(param, dic_params=dic_params, n_samples=100000)
    entropy_empirical = beta_posterior.calculate_entropy(param, dic_params=dic_params, empirical_entropy=True, n_samples=100000)

    assert(entropy_emp1.shape == (3,)), "The entropy should be a vector of size 2"
    assert(entropy_empirical.shape == (3,)), "The entropy should be a vector of size 2"
    assert entropy_emp1[0] == 0.0, "The entropy should be 0"
    assert torch.allclose(entropy_emp1, entropy_empirical, rtol=1e-1), f"The entropy should be the same kl1 {entropy_emp1}, kl2 {entropy_empirical}"
