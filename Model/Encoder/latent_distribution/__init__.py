from .gaussian import GaussianPosterior
from .uniform import UniformPosterior
from .beta import BetaPosterior


def get_latent_distribution(distribution_name, cfg):
    if distribution_name == 'gaussian':
        return GaussianPosterior(cfg)
    elif distribution_name == 'gaussian_cylindric':
        raise NotImplementedError
        # return GaussianCylindricPosterior(cfg)
    elif distribution_name == 'uniform':
        return UniformPosterior(cfg)
    elif distribution_name == 'beta':
        return BetaPosterior(cfg)
    else:
        raise NotImplementedError
