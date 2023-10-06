from .mixture_of_gaussian import MixtureOfGaussian
from .gaussian_prior import GaussianPrior

def get_prior(cfg):
    if cfg.prior.prior_name == 'mixture_of_gaussian':
        return MixtureOfGaussian(cfg)
    elif cfg.prior.prior_name == 'gaussian':
        return GaussianPrior(cfg)
    else:
        raise NotImplementedError