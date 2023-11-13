from .beta import BetaPosterior
from .gaussian import GaussianPosterior
from .gaussian_symmetrical import GaussiansymmetricalPosterior
from .uniform import UniformPosterior
from .von_mises_fischer import VonMisesFischerPosterior


def get_latent_distribution(distribution_name, cfg):
    if distribution_name == "gaussian":
        return GaussianPosterior(cfg)
    elif distribution_name == "gaussian_symmetrical":
        return GaussiansymmetricalPosterior(cfg)
    elif distribution_name == "gaussian_cylindric":
        raise NotImplementedError
        # return GaussianCylindricPosterior(cfg)
    elif distribution_name == "uniform":
        return UniformPosterior(cfg)
    elif distribution_name == "beta":
        return BetaPosterior(cfg)
    elif distribution_name == "von_mises_fischer":
        return VonMisesFischerPosterior(cfg)
    else:
        raise NotImplementedError
