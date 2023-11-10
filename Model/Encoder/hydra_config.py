import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union



@dataclass
class BaseEncoderConfig:
    network_name: str = MISSING
    activation_name: Optional[Union[str,None]] = None
    nef: int = MISSING
    latent_distribution_name: str = MISSING
    forced_latent_min: Optional[Union[float, None]]= None # Just used in Uniform
    forced_latent_max: Optional[Union[float, None]]= None # Just used in Uniform
    sigmoid_version: Optional[bool] = False
    init_network_type: Optional[Union[str, None]] = "mean"


def store_base_encoder(cs: ConfigStore):
    cs.store(group="encoder", name="base_encoder", node=BaseEncoderConfig)