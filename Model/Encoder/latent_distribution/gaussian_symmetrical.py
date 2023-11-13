import math

import torch
import torch.nn as nn

from .abstract_latent_distribution import AbstractLatentDistribution


class GaussiansymmetricalPosterior(AbstractLatentDistribution):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.cfg = cfg
        # self.mu = nn.Parameter(torch.zeros((self.cfg.trainer.nz), device=self.cfg.trainer.device), requires_grad=False)
        # self.std = nn.Parameter(torch.ones((self.cfg.trainer.nz), device=self.cfg.trainer.device), requires_grad=False)
        self.lambda_nz = lambda nz: 2 * nz

    def get_params(
        self,
        param,
    ):
        # dic_params["mu"], dic_params["log_var"] = param.chunk(2, dim=1)
        dic_params = {}
        dic_params_feedback = {}
        mu, log_var = param.chunk(2, dim=1)
        dic_params["mu"] = mu
        dic_params["log_var"] = log_var
        dic_params_feedback["||mu_encoder||"] = mu.norm(dim=1)
        dic_params_feedback["var_encoder_mean"] = log_var.exp().mean(dim=1)
        return dic_params, dic_params_feedback

    def r_sample(
        self,
        params,
        n_samples=1,
        dic_params=None,
    ):
        if dic_params is None:
            mu, log_var = params.chunk(2, dim=1)
        else:
            mu, log_var = dic_params["mu"], dic_params["log_var"]
        mu_expanded = mu.unsqueeze(0).expand(n_samples, *mu.shape)
        log_var_expanded = log_var.unsqueeze(0).expand(n_samples, *log_var.shape)
        epsilon = torch.randn_like(mu_expanded)
        shape = (mu.shape[0], *[1 for _ in range(len(mu.shape) - 1)])
        bernoulli = torch.bernoulli(torch.ones(shape) * 0.5).expand_as(mu_expanded).to(mu.device)
        return (2 * bernoulli - 1) * mu_expanded + torch.exp(log_var_expanded / 2) * epsilon

    def get_distribution(self, params, dic_params=None):
        if dic_params is None:
            mu, log_var = params.chunk(2, dim=1)
        else:
            mu, log_var = dic_params["mu"], dic_params["log_var"]

        mu_total = torch.cat([mu.unsqueeze(1), -mu.unsqueeze(1)], dim=1)
        log_var_total = torch.cat([log_var.unsqueeze(1), log_var.unsqueeze(1)], dim=1)
        encoder_distrib = torch.distributions.Normal(mu_total, torch.exp(log_var_total / 2))

        mix_dist = torch.distributions.categorical.Categorical(torch.full((mu.shape[0],2),0.5).to(mu.device))
        comp_dist = torch.distributions.Independent(encoder_distrib, 1)
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix_dist, comp_dist)

        return gmm

    def calculate_kl(self, prior, params, samples, dic_params=None, empirical_kl=False):
        if dic_params is None:
            mu_q, log_var_q = params.chunk(2, dim=1)
        else:
            mu_q, log_var_q = dic_params["mu"], dic_params["log_var"]

            # Empirical KL
        log_prob_posterior = self.log_prob(params, samples, dic_params=dic_params).reshape(-1, params.shape[0])
        log_prob_prior = prior.log_prob(samples).reshape(-1, params.shape[0])
        kl = log_prob_posterior - log_prob_prior
        return kl.mean(0)

    def get_plots(self, params, dic_params=None):
        if dic_params is None:
            mu, log_var = params.chunk(2, dim=1)
        else:
            mu, log_var = dic_params["mu"], dic_params["log_var"]
        return mu, log_var

    def calculate_entropy(
        self, params, dic_params=None, empirical_entropy=False, n_samples=100
    ):
        if dic_params is None:
            mu, log_var = params.chunk(2, dim=1)
        else:
            mu, log_var = dic_params["mu"], dic_params["log_var"]
        samples = self.r_sample(params, n_samples=n_samples, dic_params=dic_params)
        return -self.log_prob(params, samples, dic_params=dic_params).reshape(-1, params.shape[0]).mean(0)

    def log_prob(self, params, z_q, dic_params = None):
        if z_q.shape[0] == params.shape[0] and len(z_q.shape) == len(params.shape):
            return self.get_distribution(params, dic_params=dic_params).log_prob(z_q).reshape(params.shape[0],)
        else :
            z_q = z_q.reshape(-1, params.shape[0], self.cfg.trainer.nz)
            dist = self.get_distribution(params, dic_params=dic_params)
            log_prob = dist.log_prob(z_q).reshape(-1, params.shape[0],)
            return log_prob

    