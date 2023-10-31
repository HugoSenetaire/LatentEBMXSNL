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

from Model.Encoder.latent_distribution.uniform import UniformPosterior
from Model.Prior.uniform_prior import UniformPrior



def test_kl_uniform():
    cfg = get_config(config_name="conf_uniform")
    cfg.prior.prior_name = 'uniform'
    cfg.prior.min = -1.0
    cfg.prior.max = 1.0
    uniform_posterior = get_latent_distribution(cfg.encoder.latent_distribution_name, cfg)
    uniform_prior = get_prior(cfg.trainer.nz, cfg.prior)
    assert isinstance(uniform_posterior, UniformPosterior), "The uniform posterior should be a UniformPosterior"
    assert isinstance(uniform_prior, UniformPrior), "The uniform prior should be a UniformPrior"

    param = torch.tensor([[-1.0, -1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0]], dtype=torch.float32)
    dic_params, dic_params_feedback = uniform_posterior.get_params(param)

    #Verify the obtention of parameters
    assert dic_params_feedback["min_avg"].shape == (2,), "The min should be a vector of size 2"
    assert dic_params_feedback["min_avg"][0].item() == -1.0, "The min should be -1"
    assert dic_params_feedback["max_avg"][0].item() == 0.5, "The max should be 1"
    assert dic_params_feedback["min_avg"][1].item() == -1.0, "The min should be -1"
    assert dic_params_feedback["max_avg"][1].item() == 1.0, "The max should be 1"


    assert torch.allclose(dic_params["min"], torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])), "The min should be -1"
    assert torch.allclose(dic_params["max"], torch.tensor([[0.0, 1.0], [1.0, 1.0]])), "The max should be 1"

    #Verify log_prob
    assert torch.allclose(uniform_posterior.log_prob(param, param.chunk(2,1)[0]), torch.tensor([math.log(0.5), math.log(0.25)])), "The log prob is weird"
    param_expanded = param.unsqueeze(0).expand(2, -1, -1).flatten(0, 1)
    assert param_expanded.shape == (4, 4), "The shape should be (4, 4)"
    assert uniform_posterior.log_prob(param, param_expanded.chunk(2,1)[0]).shape == (2,2), "The shape should be (2,2)"
    assert torch.allclose(uniform_posterior.log_prob(param, param_expanded.chunk(2,1)[0]).flatten(), torch.tensor([math.log(0.5), math.log(0.25), math.log(0.5), math.log(0.25)])), "The log prob is weird"

    #Verify sample
    samples_posterior = uniform_posterior.r_sample(param, 1000, dic_params=dic_params)
    assert torch.all(samples_posterior >= cfg.prior.min) and torch.all(samples_posterior <= cfg.prior.max), "The samples should be in the range of [min, max]"
    assert samples_posterior.shape == (1000, 2, 2), "The samples should be of size (1000, 2)"

    # Test for the KL
    # Backward KL :
    kl_formula = uniform_posterior.calculate_kl(uniform_prior, param, samples_posterior.flatten(0,1), dic_params=dic_params)
    kl_empirical = uniform_posterior.calculate_kl(uniform_prior, param, samples_posterior.flatten(0,1), dic_params=dic_params, empirical_kl=True)
    
    assert(kl_formula.shape == (2,)), "The kl should be a vector of size 2"
    assert(kl_empirical.shape == (2,)), "The kl should be a vector of size 2"
    assert torch.allclose(kl_formula, kl_empirical, rtol=1e-3), "The kl should be the same"

    # Forward KL :
    # Not applicable for uniform distribution

    # Test entropy :
    entropy_formula = uniform_posterior.calculate_entropy(param, dic_params=dic_params)
    entropy_empirical = uniform_posterior.calculate_entropy(param, dic_params=dic_params, empirical_entropy=True)

    assert(entropy_formula.shape == (2,)), "The entropy should be a vector of size 2"
    assert(entropy_empirical.shape == (2,)), "The entropy should be a vector of size 2"
    assert torch.allclose(entropy_formula, entropy_empirical, rtol=1e-3), "The entropy should be the same"
