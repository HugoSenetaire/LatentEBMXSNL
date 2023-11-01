import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union




@dataclass
class BasePriorConfig:
    prior_name: str = MISSING

@dataclass
class GaussianPriorConfig(BasePriorConfig):
    prior_name: str = "gaussian"
    mu : Optional[float] = 0.
    sigma : Optional[float] = 1.


@dataclass
class UniformPriorConfig(BasePriorConfig):
    prior_name: str = "uniform"
    min: float = MISSING
    max: float = MISSING

@dataclass
class MixtureOfGaussianPriorConfig(BasePriorConfig):
    prior_name: str = "mixture_of_gaussian"
    nb_mixture: int = MISSING

class HyperSphericalUniformPriorConfig(BasePriorConfig):
    prior_name: str = "hyperspherical_uniform"


def store_base_prior(cs: ConfigStore):
    cs.store(group="prior", name="base_prior", node=BasePriorConfig)
    cs.store(group="prior", name="base_gaussian", node=GaussianPriorConfig)
    cs.store(group="prior", name="base_mixture_of_gaussian", node=MixtureOfGaussianPriorConfig)
    cs.store(group="prior", name="base_uniform", node=UniformPriorConfig)
    cs.store(group="prior", name="base_hyperspherical_uniform", node=HyperSphericalUniformPriorConfig)

def store_base_extra_prior(cs: ConfigStore):
    cs.store(group="extra_prior", name="base_prior", node=BasePriorConfig)
    cs.store(group="extra_prior", name="base_gaussian", node=GaussianPriorConfig)
    cs.store(group="extra_prior", name="base_mixture_of_gaussian", node=MixtureOfGaussianPriorConfig)
    cs.store(group="extra_prior", name="base_uniform", node=UniformPriorConfig)
    cs.store(group="extra_prior", name="base_hyperspherical_uniform", node=HyperSphericalUniformPriorConfig)