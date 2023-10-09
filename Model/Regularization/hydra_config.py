import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
from omegaconf import DictConfig


@dataclass
class BaseRegularizationConfig:
    coef_regul: Optional[float] = 0.0
    l2_grad: Optional[float] = 0.0
    l2_output: Optional[float] = 0.0
    l2_param: Optional[float] = 0.0
    normalize_sample_grad: Optional[bool] = False


    
def store_base_regularization(cs: ConfigStore):
    cs.store(group="regularization", name="base_config_regularization", node=BaseRegularizationConfig)



@dataclass
class BaseRegularizationEncoderConfig:
    l2_mu: float= MISSING

def store_base_regularization_encoder(cs: ConfigStore):
    cs.store(group="regularization_encoder", name="base_config_regularization_encoder", node=BaseRegularizationEncoderConfig)