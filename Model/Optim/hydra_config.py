import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union



@dataclass
class BaseOptimConfig:
    optimizer_name: str = MISSING
    clip_grad_value: Optional[float] = None
    clip_grad_type: Optional[Union[str, None]] = "norm"  # norm, value, adam, none
    nb_sigma: Optional[float] = 3.0

@dataclass
class AdamOptimConfig(BaseOptimConfig):
    optimizer_name: str = "adam"
    lr: float = MISSING
    weight_decay: float = MISSING
    b1: float = MISSING
    b2: float = MISSING
    eps: float = MISSING

@dataclass
class AdamwOptimConfig(BaseOptimConfig):
    optimizer_name: str = "adamw"
    lr: float = MISSING
    weight_decay: float = MISSING
    b1: float = MISSING
    b2: float = MISSING
    eps: float = MISSING

def store_base_optim(cs: ConfigStore):
    cs.store(group="optim_energy", name="base_optim", node=BaseOptimConfig)
    cs.store(group="optim_prior", name="base_optim", node=BaseOptimConfig)
    cs.store(group="optim_generator", name="base_optim", node=BaseOptimConfig)
    cs.store(group="optim_encoder", name="base_optim", node=BaseOptimConfig)

    cs.store(group="optim_energy", name="adam_base", node=AdamOptimConfig)
    cs.store(group="optim_prior", name="adam_base", node=AdamOptimConfig)
    cs.store(group="optim_generator", name="adam_base", node=AdamOptimConfig)
    cs.store(group="optim_encoder", name="adam_base", node=AdamOptimConfig)

    cs.store(group="optim_energy", name="adamw_base", node=AdamwOptimConfig)
    cs.store(group="optim_prior", name="adamw_base", node=AdamwOptimConfig)
    cs.store(group="optim_generator", name="adamw_base", node=AdamwOptimConfig)
    cs.store(group="optim_encoder", name="adamw_base", node=AdamwOptimConfig)