import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union




@dataclass
class BasePriorConfig:
    prior_name: str = "gaussian"
   
@dataclass
class MixtureOfGaussianPriorConfig(BasePriorConfig):
    prior_name: str = "mixture_of_gaussian"
    nb_mixture: int = MISSING


def store_base_prior(cs: ConfigStore):
    cs.store(group="prior", name="base_prior", node=BasePriorConfig)
    cs.store(group="prior", name="mixture_of_gaussian", node=MixtureOfGaussianPriorConfig)

