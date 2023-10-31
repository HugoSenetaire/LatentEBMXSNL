import torch

class AggregatePosterior():
    """
    Aggregate posterior for the VAE. 
    We create a mixture of the encoder distribution.
    """
    def __init__(self, encoder_distrib, nb_components, device = "cuda"):
        self.encoder_distrib = encoder_distrib
        self.log_mix = 1/nb_components * torch.ones((nb_components,))
        component_prob = (self.log_mix - torch.logsumexp(self.log_mix, dim=0)).exp().to(device)
        
        self.mix_dist = torch.distributions.categorical.Categorical(component_prob)
        self.comp_dist = torch.distributions.Independent(self.encoder_distrib,1)
        self.gmm = torch.distributions.mixture_same_family.MixtureSameFamily(self.mix_dist, self.comp_dist)
       

    def log_prob(self, z):
        return self.gmm.log_prob(z).reshape(z.shape[0])
    
    def sample(self, n):
        return self.gmm.sample((n,))

