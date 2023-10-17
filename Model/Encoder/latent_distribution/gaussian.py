

import torch
import torch.nn as nn
import math 

class GaussianPosterior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.mu = nn.Parameter(torch.zeros((self.cfg.trainer.nz), device=self.cfg.trainer.device), requires_grad=False)
        # self.std = nn.Parameter(torch.ones((self.cfg.trainer.nz), device=self.cfg.trainer.device), requires_grad=False)


    def r_sample(self, params, dic_params = None, n_samples=1):
        if dic_params is None :
            mu, log_var = params.chunk(2, dim=1)
        else :
            mu, log_var = dic_params["mu"], dic_params["log_var"]
        mu_expanded = mu.unsqueeze(0).expand(n_samples, *mu.shape)
        log_var_expanded = log_var.unsqueeze(0).expand(n_samples, *log_var.shape)
        epsilon = torch.randn_like(mu_expanded)
        return mu_expanded + torch.exp(log_var_expanded/2) * epsilon
    
    def get_params(self, param,):
        # dic_params["mu"], dic_params["log_var"] = param.chunk(2, dim=1)
        dic_params = {}
        dic_params_feedback = {}
        mu, log_var = param.chunk(2, dim=1)
        dic_params["mu"]= mu
        dic_params["log_var"]= log_var
        dic_params_feedback["||mu||"]= mu.norm(dim=1)
        dic_params_feedback["||log_var||"]= log_var.norm(dim=1)
        return dic_params, dic_params_feedback


    def get_distribution(self, params, dic_params = None):
        if dic_params is None :
            mu, log_var = params.chunk(2, dim=1)
        else :
            mu, log_var = dic_params["mu"], dic_params["log_var"]
        return torch.distributions.Normal(mu, torch.exp(log_var/2))

    
    def calculate_kl(self, prior, params, samples, dic_params = None, empirical_kl = False):
        if dic_params is None :
            mu_q, log_var_q = params.chunk(2, dim=1)
        else :
            mu_q, log_var_q = dic_params["mu"], dic_params["log_var"]
        if self.cfg.prior.prior_name == 'gaussian' and not empirical_kl:
            mu_q, log_var_q = params.chunk(2, dim=1)
            mu_prior, log_var_prior = prior.mu.unsqueeze(0), prior.log_var.unsqueeze(0)
            kl_loss = 0.5 * (log_var_prior - log_var_q + (torch.exp(log_var_q) + (mu_q - mu_prior).pow(2)) / torch.exp(log_var_prior) - 1)
            kl_loss = kl_loss.reshape(samples.shape[0], self.cfg.trainer.nz).sum(1)
            return kl_loss
        else :
            # Empirical KL
            return self.log_prob(params, samples) - prior.log_prob(samples)

    def calculate_entropy(self, params, dic_params = None, empirical_entropy = False):
        if dic_params is None :
            mu, log_var = params.chunk(2, dim=1)
        else :
            mu, log_var = dic_params["mu"], dic_params["log_var"]
        entropy_posterior = 0.5 * (1 + log_var + math.log(2 * math.pi))
        return entropy_posterior.reshape(params.shape[0], self.cfg.trainer.nz).sum(1)


    def log_prob(self, params, x_hat, dic_params = None):
        '''
        Get the log probability of the given x_hat given the parameters of the distribution.
        Handle also for multiple inputs
        :param params: The parameters of the distribution
        :param x_hat: The sample to evaluate
        '''
        if dic_params is None :
            mu, log_var = params.chunk(2, dim=1)
        else :
            mu, log_var = dic_params["mu"], dic_params["log_var"]
        if x_hat.shape[0] == mu.shape[0]:
            return torch.distributions.Normal(mu, torch.exp(log_var/2)).log_prob(x_hat).reshape(mu.shape[0], self.cfg.trainer.nz).sum(1)
        else :
            x_hat = x_hat.reshape(-1, *mu.shape)
            return torch.distributions.Normal(mu, torch.exp(log_var/2)).log_prob(x_hat).reshape(x_hat.shape[0], mu.shape[0], self.cfg.trainer.nz).sum(2)



# class GaussianCylindricPosterior(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg


#     def get_params(self, params):
#         mu_cylindric, log_var_cylindric = params.chunk(2, dim=1)
#         mu = torch.cos(mu_cylindric)
#         log_var = log_var_cylindric
#         return torch.cat((mu, log_var), dim=1)

#     def r_sample(self, params, n_samples=1):
#         mu_cylindric, log_var_cylindric = params.chunk(2, dim=1)
#         mu_expanded = mu.unsqueeze(0).expand(n_samples, *mu.shape)
#         log_var_expanded = log_var.unsqueeze(0).expand(n_samples, *log_var.shape)
#         epsilon = torch.randn_like(mu_expanded)
#         return mu_expanded + torch.exp(log_var_expanded/2) * epsilon


#     def log_prob(self, params, x_hat):
#         '''
#         Get the log probability of the given x_hat given the parameters of the distribution.
#         Handle also for multiple inputs
#         :param params: The parameters of the distribution
#         :param x_hat: The sample to evaluate
#         '''
#         mu, log_var = params.chunk(2, dim=1)
#         if x_hat.shape[0] == mu.shape[0]:
#             return torch.distributions.Normal(mu, torch.exp(log_var/2)).log_prob(x_hat).reshape(mu.shape[0], self.cfg.trainer.nz).sum(1)
#         else :
#             mu_expanded = mu.unsqueeze(0).expand(x_hat.shape[0], *mu.shape)
#             log_var_expanded = log_var.unsqueeze(0).expand(x_hat.shape[0], *log_var.shape)
#             return torch.distributions.Normal(mu_expanded, torch.exp(log_var_expanded/2)).log_prob(x_hat).reshape(x_hat.shape[0], mu.shape[0], self.cfg.trainer.nz).sum(2)

