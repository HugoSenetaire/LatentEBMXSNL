import torch.nn as nn
import torch

class BetaPosterior(nn.Module):
    def __init__(self, cfg):
        super(BetaPosterior, self).__init__()
         
        self.cfg = cfg
        if cfg.prior.prior_name == 'uniform':
            self.forced_min = cfg.prior.min
            self.forced_max = cfg.prior.max
        else :
            self.forced_min = -1.
            self.forced_max = 1.



    def calculate_kl(self, prior, params, samples, dic_params = None, empirical_kl = False):
        # Empirical KL 
        log_prob_posterior = self.log_prob(params, samples, dic_params=dic_params).reshape(-1, params.shape[0])
        log_prob_prior = prior.log_prob(samples).reshape(-1, params.shape[0])
        kl = log_prob_posterior - log_prob_prior
        return kl.mean(0)
        
        
    def calculate_entropy(self, params, dic_params = None, empirical_entropy = False, n_samples=100):
        # Empirical entropy
        samples = self.r_sample(params, n_samples=n_samples, dic_params=dic_params)
        return -self.log_prob(params, samples, dic_params=dic_params).mean(0)
    

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
    
    def r_sample(self, params, n_samples=1, dic_params = None):
        samples = self.get_distribution(params, dic_params = dic_params).rsample((n_samples,))
        return samples

    def log_prob(self, params, z_q, dic_params = None):
        if z_q.shape[0] == params.shape[0]:
            return self.get_distribution(params, dic_params=dic_params).log_prob(z_q).reshape(params.shape[0], self.cfg.trainer.nz).sum(1)
        else :
            z_q = z_q.reshape(-1, params.shape[0], self.cfg.trainer.nz)
            dist = self.get_distribution(params, dic_params=dic_params)
            log_prob = dist.log_prob(z_q).reshape(-1, params.shape[0], self.cfg.trainer.nz).sum(2)
            return log_prob

    
