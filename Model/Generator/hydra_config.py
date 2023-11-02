import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union



@dataclass
class BaseGeneratorConfig:
    network_name: str = MISSING
    activation_name: Optional[Union[str,None]] = None
    loss_reconstruction_name: str = MISSING
    llhd_sigma: Optional[Union[float,None]] = None
    ngf: int = MISSING

    def __post_init__(self):
        self.network_name = self.network_name.lower()
        self.activation_name = self.activation_name.lower()
        self.generator.get_loss_name = self.generator.get_loss_name.lower()
        if self.generator.get_loss_name == "gaussian":
            assert self.llhd_sigma is not None, "llhd_sigma must be specified for gaussian loss"




def store_base_generator(cs: ConfigStore):
    cs.store(group="generator", name="base_generator", node=BaseGeneratorConfig)