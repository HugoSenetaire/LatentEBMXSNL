import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union



@dataclass
class BaseEnergyConfig:
    network_name: str = MISSING
    activation_name: Union[str,None] = MISSING
    ndf: int = MISSING


def store_base_energy(cs: ConfigStore):
    cs.store(group="energy", name="base_energy", node=BaseEnergyConfig)