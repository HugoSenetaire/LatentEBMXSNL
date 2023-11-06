import torch.nn as nn
import torch
import torch.nn.functional as F
from ...Utils.hyperspherical_utils import VonMisesFisher
from .abstract_latent_distribution import AbstractLatentDistribution
import math 

class VonMisesFischerPosterior(AbstractLatentDistribution):
    def __init__(self, cfg):
        super(VonMisesFischerPosterior, self).__init__(cfg=cfg)
        self.cfg = cfg
        self.lambda_nz = lambda nz : nz+1


    def get_params(self, params):
        dic_params = {}
        dic_params_feedback = {}
        
        # compute mean and concentration of the von Mises-Fisher
        z_mean = params[:, :-1]
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        # the `+ 1` prevent collapsing behaviors
        z_var = params[:, -1:]
        # z_var = F.softplus(z_var) + 1
        z_var = torch.sigmoid(z_var)*(2*math.pi-1)+1


        dic_params["z_mean"] = z_mean
        dic_params["z_var"] = z_var

        dic_params_feedback["z_mean_avg"]= z_mean.mean(dim=1)
        dic_params_feedback["z_var_avg"]= z_var.mean(dim=1)

        return dic_params, dic_params_feedback



    def get_distribution(self, params, dic_params = None):
        if dic_params is None :
            dic_params, _ = self.get_params(params,)
        z_mean, z_var = dic_params["z_mean"], dic_params["z_var"]
        return VonMisesFisher(z_mean, z_var)


    def get_plots(self, params, dic_params = None):
        if dic_params is None :
            dic_params, _ = self.get_params(params)
        z_mean, z_var = dic_params["z_mean"], dic_params["z_var"]
        mu = z_mean
        log_var = torch.log(z_var)
        return mu, log_var
    
  

    def calculate_kl(self, prior, params, samples, dic_params = None, empirical_kl = False):
        return super().calculate_kl(prior, params, samples, dic_params = dic_params, empirical_kl = True)
        
    def calculate_entropy(self, params, dic_params = None, empirical_entropy = False, n_samples=100):
        return super().calculate_entropy(params, dic_params = dic_params, empirical_entropy = True, n_samples=n_samples)

    def log_prob(self, params, z_q, dic_params = None):
        """
        Main difference with the other log prob is that I get a log likelihood not per dimension.
        """
        if z_q.shape[0] == params.shape[0]:
            return self.get_distribution(params, dic_params=dic_params).log_prob(z_q).reshape(params.shape[0],)
        else :
            z_q = z_q.reshape(-1, params.shape[0], self.cfg.trainer.nz)
            dist = self.get_distribution(params, dic_params=dic_params)
            log_prob = dist.log_prob(z_q).reshape(-1, params.shape[0],)
            return log_prob