import torch.nn as nn
import torch

class AbstractLatentDistribution(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # How much I need to expand the latent dimension to get all parameters
        self.lambda_nz = None

    
    def calculate_kl(self, prior, params, samples, dic_params = None, empirical_kl = False):
        if empirical_kl:
            # Empirical KL             
            log_prob_posterior = self.log_prob(params, samples, dic_params=dic_params).reshape(-1, params.shape[0])
            log_prob_prior = prior.log_prob(samples).reshape(-1, params.shape[0])
            kl = log_prob_posterior - log_prob_prior
            return kl.mean(0)   
        else :
            raise NotImplementedError("KL not implemented for this distribution")
        
        
    def calculate_entropy(self, params, dic_params = None, empirical_entropy = False, n_samples=100):
        if empirical_entropy :
        # Empirical entropy
            samples = self.r_sample(params, n_samples=n_samples, dic_params=dic_params)
            return -self.log_prob(params, samples, dic_params=dic_params).mean(0)
        else :
            raise NotImplementedError("Entropy not implemented for this distribution")
    

    def get_params(self, params):
        """
        Return a dictionnary with the obtained parameters and another with feedback to log
        """
        raise NotImplementedError("get_params not implemented for this distribution")


    def get_distribution(self, params, dic_params = None):
        raise NotImplementedError("get_distribution not implemented for this distribution")
    

    def get_plots(self, params, dic_params = None):
        """
        Return the mean and log_var for plotting
        """
    
    def r_sample(self, params, n_samples=1, dic_params = None):
        samples = self.get_distribution(params, dic_params = dic_params).rsample(torch.Size((n_samples,)))
        return samples

    def log_prob(self, params, z_q, dic_params = None):
        if z_q.shape[0] == params.shape[0]:
            return self.get_distribution(params, dic_params=dic_params).log_prob(z_q).reshape(params.shape[0], self.cfg.trainer.nz).sum(1)
        else :
            z_q = z_q.reshape(-1, params.shape[0], self.cfg.trainer.nz)
            dist = self.get_distribution(params, dic_params=dic_params)
            log_prob = dist.log_prob(z_q).reshape(-1, params.shape[0], self.cfg.trainer.nz).sum(2)
            return log_prob

    
