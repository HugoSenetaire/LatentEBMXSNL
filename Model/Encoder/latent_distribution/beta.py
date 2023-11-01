import torch.nn as nn
import torch
from .abstract_latent_distribution import AbstractLatentDistribution


class BetaPosterior(AbstractLatentDistribution):
    def __init__(self, cfg):
        super(BetaPosterior, self).__init__(cfg=cfg)
        if cfg.prior.prior_name == 'uniform':
            self.forced_min = cfg.prior.min
            self.forced_max = cfg.prior.max
        else :
            self.forced_min = -1.
            self.forced_max = 1.

        # How much I need to expand the latent dimension to get all parameters
        self.lambda_nz = lambda nz : 2*nz



    def calculate_kl(self, prior, params, samples, dic_params = None, empirical_kl = False):
        return super().calculate_kl(prior, params, samples, dic_params = dic_params, empirical_kl = True)
        
        
    def calculate_entropy(self, params, dic_params = None, empirical_entropy = False, n_samples=100):
        return super().calculate_entropy(params, dic_params = dic_params, empirical_entropy = True, n_samples=n_samples)
    
    def get_params(self, params):
        dic_params = {}
        dic_params_feedback = {}

        alpha_log, beta_log = params.chunk(2, dim=1)
        alpha = torch.exp(alpha_log)
        beta = torch.exp(beta_log)

        dic_params["alpha"] = alpha
        dic_params["beta"] = beta
      
        dic_params_feedback["alpha_avg"]= alpha.mean(dim=1)
        dic_params_feedback["beta_avg"]= beta.mean(dim=1)

        return dic_params, dic_params_feedback


    def get_distribution(self, params, dic_params = None):
        if dic_params is None :
            dic_params, _ = self.get_params(params,)
        alpha, beta = dic_params["alpha"], dic_params["beta"]
        return torch.distributions.Beta(concentration0=alpha, concentration1=beta)
    

    def get_plots(self, params, dic_params = None):
        if dic_params is None :
            dic_params, _ = self.get_params(params)
        alpha, beta = dic_params["alpha"], dic_params["beta"]
        mu = (alpha)/(alpha+beta)
        log_var = torch.log((alpha*beta)/((alpha+beta)**2*(alpha+beta+1)))
        return mu, log_var
  
