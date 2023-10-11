import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union



@dataclass
class BaseSchedulerConfig:
    scheduler_name: str = MISSING

@dataclass
class ExponentialLRSchedulerConfig(BaseSchedulerConfig):
    scheduler_name: str = "exponential_lr"
    gamma: float = MISSING


def store_base_scheduler(cs: ConfigStore):
    cs.store(group="scheduler_energy", name="base_scheduler", node=BaseSchedulerConfig)
    cs.store(group="scheduler_prior", name="base_scheduler", node=BaseSchedulerConfig)
    cs.store(group="scheduler_generator", name="base_scheduler", node=BaseSchedulerConfig)
    cs.store(group="scheduler_encoder", name="base_scheduler", node=BaseSchedulerConfig)

    cs.store(group="scheduler_energy", name="base_exponential_lr_scheduler", node=ExponentialLRSchedulerConfig)
    cs.store(group="scheduler_prior", name="base_exponential_lr_scheduler", node=ExponentialLRSchedulerConfig)
    cs.store(group="scheduler_generator", name="base_exponential_lr_scheduler", node=ExponentialLRSchedulerConfig)
    cs.store(group="scheduler_encoder", name="base_exponential_lr_scheduler", node=ExponentialLRSchedulerConfig)
