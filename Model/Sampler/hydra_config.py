import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
from omegaconf import DictConfig

@dataclass
class BaseLangevinSampler:
    K: int = MISSING
    a: float = MISSING


def store_base_langevin_sampler(cs: ConfigStore):
    cs.store(group="sampler_prior", name="base_langevin_sampler", node=BaseLangevinSampler)
    cs.store(group="sampler_posterior", name="base_langevin_sampler", node=BaseLangevinSampler)

