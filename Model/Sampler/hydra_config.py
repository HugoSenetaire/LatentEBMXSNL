import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
from omegaconf import DictConfig

@dataclass
class BaseSampler:
    sampler_name: str = MISSING
    num_chains_test: int = MISSING # In training, defined by the batch size
    num_samples: int = MISSING
    thinning: int = MISSING
    warmup_steps: int = MISSING
    step_size: Optional[float] = 1.0
    clamp_min_data: Optional[float] = None
    clamp_max_data: Optional[float] = None
    clamp_min_grad: Optional[float] = None
    clamp_max_grad: Optional[float] = None
    clip_data_norm : Optional[float] = None
    clip_grad_norm: Optional[float] = None
    hyperspherical: Optional[bool] = False

@dataclass
class BaseLangevinSampler(BaseSampler):
    sampler_name: str = "langevin"
    num_chains_test: int = MISSING # In training, defined by the batch size
    num_samples: int = MISSING
    thinning: int = MISSING
    warmup_steps: int = MISSING
    step_size: float = MISSING
    clamp_min_data: Optional[float] = None
    clamp_max_data: Optional[float] = None
    clamp_min_grad: Optional[float] = None
    clamp_max_grad: Optional[float] = None
    clip_data_norm : Optional[float] = None
    clip_grad_norm: Optional[float] = None
    hyperspherical: Optional[bool] = False


@dataclass
class BaseNutsSampler(BaseSampler):
    sampler_name: str = "nuts"
    num_chains_test: int = MISSING # In training, defined by the batch size
    num_samples: int = MISSING
    thinning: int = MISSING
    warmup_steps: int = MISSING
    step_size: float = 1.0
    clamp_min_data: Optional[float] = None
    clamp_max_data: Optional[float] = None
    clamp_min_grad: Optional[float] = None
    clamp_max_grad: Optional[float] = None
    clip_data_norm : Optional[float] = None
    clip_grad_norm: Optional[float] = None
    hyperspherical: Optional[bool] = False
    multiprocess: str = "None" # "None" or "Cheating" or "Standard" Depending on whether I want to use multiprocessing or not
                                # In the cheating version, each batch has a single step_size, which is not exactly nuts, but it is faster
    def __post_init__(self):
        assert self.multiprocess in ["None", "Cheating", "Standard"]


def store_base_langevin_sampler(cs: ConfigStore):
    cs.store(group="sampler_prior", name="base_sampler", node=BaseSampler)
    cs.store(group="sampler_prior", name="base_langevin_sampler", node=BaseLangevinSampler)
    cs.store(group="sampler_prior", name="base_nuts_sampler", node=BaseNutsSampler)

    cs.store(group="sampler_posterior", name="base_sampler", node=BaseSampler)
    cs.store(group="sampler_posterior", name="base_langevin_sampler", node=BaseLangevinSampler)
    cs.store(group="sampler_posterior", name="base_nuts_sampler", node=BaseNutsSampler)

