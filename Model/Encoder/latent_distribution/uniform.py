import torch.nn as nn
import torch
from .abstract_latent_distribution import AbstractLatentDistribution

class UniformPosterior(AbstractLatentDistribution):
    def __init__(self, cfg):
        super(UniformPosterior, self).__init__(cfg=cfg)
         
        self.lambda_nz = lambda nz : 2*nz
        if cfg.prior.prior_name == 'uniform':
            self.forced_min = cfg.prior.min
            self.forced_max = cfg.prior.max
        else :
            self.forced_min = cfg.encoder.latent_min
            self.forced_max = cfg.encoder.latent_max
        if cfg.encoder.sigmoid_version and self.forced_max is None or self.forced_min is None:
            raise ValueError("If sigmoid version is used, forced_min and forced_max should be defined")
            
        self.sigmoid_version = self.cfg.encoder.sigmoid_version


    def calculate_kl(self, prior, params, samples, dic_params = None, empirical_kl = False):
        if self.cfg.prior.prior_name == 'uniform' and not empirical_kl:
            if dic_params is None :
                dic_params, _ = self.get_params(params)
            min_approx_post, max_approx_post = dic_params["min"], dic_params["max"]
            min_prior, max_prior = prior.min.unsqueeze(0), prior.max.unsqueeze(0)
            assert torch.all(min_prior<=min_approx_post), "The approximated posterior should be included in the prior"
            assert torch.all(max_approx_post<=max_prior), "The approximated posterior should be included in the prior"
            return torch.log(max_prior - min_prior).sum(1) - torch.log(max_approx_post - min_approx_post).sum(1)
        else :
            return super().calculate_kl(prior, params, samples, dic_params = dic_params, empirical_kl = empirical_kl)
        
    def calculate_entropy(self, params, dic_params = None, empirical_entropy = False, n_samples=100):
        if dic_params is None :
            dic_params, _ = self.get_params(params)
        min_aux, max_aux = dic_params["min"], dic_params["max"]
        if empirical_entropy:
            return super().calculate_entropy(params, dic_params = dic_params, empirical_entropy = empirical_entropy, n_samples=n_samples)
        return torch.log(max_aux - min_aux).reshape(params.shape[0], self.cfg.trainer.nz).sum(1)
    

    def get_params(self, params):
        dic_params = {}
        dic_params_feedback = {}

        if self.sigmoid_version:
            logit_min, logit_max = params.chunk(2, dim=1)
            min_aux = torch.sigmoid(logit_min)*(self.forced_max - self.forced_min)+self.forced_min
            max_aux = torch.sigmoid(logit_max)*(self.forced_max - min_aux)+min_aux
        else:
            min_aux, log_add_aux = params.chunk(2, dim=1)
            min_aux = min_aux.clamp(self.forced_min, self.forced_max-1e-6)
            max_aux= (min_aux + log_add_aux.exp()).clamp(self.forced_min, self.forced_max)

        dic_params["min"] = min_aux
        dic_params["max"] = max_aux

        dic_params_feedback["min_avg"]= min_aux.mean(dim=1)
        dic_params_feedback["max_avg"]= max_aux.mean(dim=1)

        return dic_params, dic_params_feedback


    def get_distribution(self, params, dic_params = None):
        if dic_params is None :
            dic_params, _ = self.get_params(params,)
        min_aux, max_aux = dic_params["min"], dic_params["max"]
        return torch.distributions.Uniform(min_aux, max_aux)


    def get_plots(self, params, dic_params = None):
        if dic_params is None :
            dic_params, _ = self.get_params(params)
        min_aux, max_aux = dic_params["min"], dic_params["max"]
        mu = (min_aux + max_aux)/2
        log_var = torch.log((max_aux - min_aux)**2/12)
        return mu, log_var
    