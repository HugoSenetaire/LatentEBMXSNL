import sys
import os
import torch
import math
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
current_path.split('L')[0]
sys.path.append(current_path.split('Model')[0])
from Model.Tests.utils_test import get_config

from Model.Encoder.latent_distribution import get_latent_distribution
from Model.Prior import get_prior

from Model.Encoder.latent_distribution.gaussian import GaussianPosterior
from Model.Prior.gaussian_prior import GaussianPrior



def test_gaussian():
    cfg = get_config(config_name="conf")
    gaussian_posterior = get_latent_distribution(cfg.encoder.latent_distribution_name, cfg)
    uniform_prior = get_prior(cfg.trainer.nz, cfg.prior)
    assert isinstance(gaussian_posterior, GaussianPosterior), "The uniform posterior should be a UniformPosterior"
    assert isinstance(uniform_prior, GaussianPrior), "The uniform prior should be a GaussianPrior"

    param = torch.tensor([[0.0, 1.0, 0.0, 0.0], [-1.0, -1.0, 0.0, 0.0]], dtype=torch.float32)
    dic_params, dic_params_feedback = gaussian_posterior.get_params(param)


    #Verify the obtention of parameters
    assert dic_params_feedback["||mu_encoder||"].shape == (2,), "Shape should be (2,)"
    assert dic_params_feedback["||mu_encoder||"][0].item() == math.sqrt(1), "The norm should be 1"
    assert dic_params_feedback["log_var_encoder_mean"][0].item() == 0., "The variance should be 1"
    assert np.absolute(dic_params_feedback["||mu_encoder||"][1].item()- math.sqrt(2))<1e-1, "The norm should be sqrt(2)"
    assert dic_params_feedback["log_var_encoder_mean"][1].item() == 0.0, "The variance should be 1"


    assert torch.allclose(dic_params["mu"], torch.tensor([[0.0, 1.0], [-1.0, -1.0]])), "Mu is not correct"
    assert torch.allclose(dic_params["log_var"], torch.tensor([[0.0, 0.0], [0.0, 0.0]])), "log var is not correct"

    #Verify log_prob
    dist1 = torch.distributions.Normal(0,1)
    dist2 = torch.distributions.Normal(1,1)
    dist3 = torch.distributions.Normal(-1,1)
    targets = [dist1.log_prob(torch.tensor(0.,))+ dist2.log_prob(torch.tensor(1.,)), 2* dist3.log_prob(torch.tensor(-1.,))]
    assert torch.allclose(gaussian_posterior.log_prob(param, param.chunk(2,1)[0]), torch.stack(targets)), "The log prob is weird"
    param_expanded = param.unsqueeze(0).expand(2, -1, -1).flatten(0, 1)
    assert param_expanded.shape == (4, 4), "The shape should be (4, 4)"
    assert gaussian_posterior.log_prob(param, param_expanded.chunk(2,1)[0]).shape == (2,2), "The shape should be (2,2)"
    assert torch.allclose(gaussian_posterior.log_prob(param, param_expanded.chunk(2,1)[0]).flatten(), torch.stack(targets*2)), "The log prob is weird"

    #Verify sample
    samples_posterior = gaussian_posterior.r_sample(param, n_samples=10000, dic_params=dic_params)
    assert samples_posterior.shape == (10000, 2, 2), "The samples should be of size (1000, 2)"

    # Test for the KL
    # Backward KL :
    kl_formula = gaussian_posterior.calculate_kl(uniform_prior, param, samples_posterior.flatten(0,1), dic_params=dic_params)
    kl_empirical = gaussian_posterior.calculate_kl(uniform_prior, param, samples_posterior.flatten(0,1), dic_params=dic_params, empirical_kl=True)
    
    assert(kl_formula.shape == (2,)), "The kl should be a vector of size 2"
    assert(kl_empirical.shape == (2,)), "The kl should be a vector of size 2"
    assert torch.allclose(kl_formula, kl_empirical, rtol=1e-1), f"The kl should be the same, empirical : {kl_empirical}  formula :{kl_formula}"

    # Forward KL :
    

    # Test entropy :
    entropy_formula = gaussian_posterior.calculate_entropy(param, dic_params=dic_params)
    entropy_empirical = gaussian_posterior.calculate_entropy(param, dic_params=dic_params, empirical_entropy=True, n_samples=100000)

    assert(entropy_formula.shape == (2,)), "The entropy should be a vector of size 2"
    assert(entropy_empirical.shape == (2,)), "The entropy should be a vector of size 2"
    assert torch.allclose(entropy_formula, entropy_empirical, rtol=1e-1), "The entropy should be the same"
