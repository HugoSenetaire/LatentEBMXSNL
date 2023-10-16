from .mixture_of_gaussian import MixtureOfGaussian
from .gaussian_prior import GaussianPrior
from .uniform_prior import UniformPrior

def get_prior(nz, cfg_prior):
    if cfg_prior.prior_name == 'mixture_of_gaussian':
        return MixtureOfGaussian(nz, cfg_prior)
    elif cfg_prior.prior_name == 'gaussian':
        return GaussianPrior(nz, cfg_prior)
    elif cfg_prior.prior_name == 'uniform':
        return UniformPrior(nz, cfg_prior)
    else:
        raise NotImplementedError
    
def get_extra_prior(nz, cfg_extra_prior):
    if cfg_extra_prior.prior_name == 'mixture_of_gaussian':
        return MixtureOfGaussian(nz, cfg_extra_prior)
    elif cfg_extra_prior.prior_name == 'gaussian':
        return GaussianPrior(nz, cfg_extra_prior)
    elif cfg_extra_prior.prior_name == 'uniform':
        return UniformPrior(nz, cfg_extra_prior)
    else:
        raise NotImplementedError
