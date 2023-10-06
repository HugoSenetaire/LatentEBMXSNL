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


def store_base_encoder(cs: ConfigStore):
    cs.store(group="encoder", name="base_encoder", node=BaseEncoderConfig)