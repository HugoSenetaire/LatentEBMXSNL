

import torch
import torch.nn as nn
import math 

class GaussianPosterior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.mu = nn.Parameter(torch.zeros((self.cfg.trainer.nz), device=self.cfg.trainer.device), requires_grad=False)
        # self.std = nn.Parameter(torch.ones((self.cfg.trainer.nz), device=self.cfg.trainer.device), requires_grad=False)


    def r_sample(self, params, n_samples=1, dic_params = None, ):
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
        dic_params_feedback["||mu_encoder||"]= mu.norm(dim=1)
        dic_params_feedback["var_encoder_mean"]= log_var.exp().mean(dim=1)
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
            kl = 0.5 * (log_var_prior - log_var_q + (torch.exp(log_var_q) + (mu_q - mu_prior).pow(2)) / torch.exp(log_var_prior) - 1)
            kl = kl.reshape(params.shape[0], self.cfg.trainer.nz).sum(1)
            return kl
        else :
            # Empirical KL
            log_prob_posterior = self.log_prob(params, samples, dic_params=dic_params).reshape(-1, params.shape[0])
            log_prob_prior = prior.log_prob(samples).reshape(-1, params.shape[0])
            kl = log_prob_posterior - log_prob_prior
            return kl.mean(0)
        
    def get_plots(self, params, dic_params = None):
        if dic_params is None :
            mu, log_var = params.chunk(2, dim=1)
        else :
            mu, log_var = dic_params["mu"], dic_params["log_var"]
        return mu, log_var


    def calculate_entropy(self, params, dic_params = None, empirical_entropy = False, n_samples=100):
        if dic_params is None :
            mu, log_var = params.chunk(2, dim=1)
        else :
            mu, log_var = dic_params["mu"], dic_params["log_var"]
        if empirical_entropy:
            samples = self.r_sample(params, n_samples=n_samples, dic_params=dic_params)
            return -self.log_prob(params, samples, dic_params=dic_params).reshape(-1, params.shape[0]).mean(0)
        
        entropy_posterior = 0.5 * (1 + log_var + math.log(2 * math.pi))
        return entropy_posterior.reshape(params.shape[0], self.cfg.trainer.nz).sum(1)


    def log_prob(self, params, z_q, dic_params = None):
        '''
        Get the log probability of the given z_q given the parameters of the distribution.
        Handle also for multiple inputs
        :param params: The parameters of the distribution
        :param z_q: The sample to evaluate
        '''
        if dic_params is None :
            mu, log_var = params.chunk(2, dim=1)
        else :
            mu, log_var = dic_params["mu"], dic_params["log_var"]
        if z_q.shape[0] == mu.shape[0]:
            return torch.distributions.Normal(mu, torch.exp(log_var/2)).log_prob(z_q).reshape(mu.shape[0], self.cfg.trainer.nz).sum(1)
        else :
            z_q = z_q.reshape(-1, *mu.shape)
            return torch.distributions.Normal(mu, torch.exp(log_var/2)).log_prob(z_q).reshape(z_q.shape[0], mu.shape[0], self.cfg.trainer.nz).sum(2)



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


#     def log_prob(self, params, z_q):
#         '''
#         Get the log probability of the given z_q given the parameters of the distribution.
#         Handle also for multiple inputs
#         :param params: The parameters of the distribution
#         :param z_q: The sample to evaluate
#         '''
#         mu, log_var = params.chunk(2, dim=1)
#         if z_q.shape[0] == mu.shape[0]:
#             return torch.distributions.Normal(mu, torch.exp(log_var/2)).log_prob(z_q).reshape(mu.shape[0], self.cfg.trainer.nz).sum(1)
#         else :
#             mu_expanded = mu.unsqueeze(0).expand(z_q.shape[0], *mu.shape)
#             log_var_expanded = log_var.unsqueeze(0).expand(z_q.shape[0], *log_var.shape)
#             return torch.distributions.Normal(mu_expanded, torch.exp(log_var_expanded/2)).log_prob(z_q).reshape(z_q.shape[0], mu.shape[0], self.cfg.trainer.nz).sum(2)


