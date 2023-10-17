from .gaussian import GaussianPosterior
from .uniform import UniformPosterior


def get_latent_distribution(distribution_name, cfg):
    if distribution_name == 'gaussian':
        return GaussianPosterior(cfg)
    elif distribution_name == 'gaussian_cylindric':
        raise NotImplementedError
        # return GaussianCylindricPosterior(cfg)
    elif distribution_name == 'uniform':
        return UniformPosterior(cfg)
    else:
        raise NotImplementedError
