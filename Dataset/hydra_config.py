import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
from omegaconf import DictConfig

@dataclass
class BaseDatasetConfig:

    # Used by the trainer to estimate the normalization constant
    dataset_name: str = MISSING

    # Iteration for training
    batch_size: int = 256
    batch_size_val: int = 256
    batch_size_test: int = 256
    root_dataset: Optional[Union[str,None]] = None

    # Dimension fixed by the dataset
    nc : int = MISSING
    img_size : int = MISSING

    # Transform back name :
    transform_back_name: str = MISSING

    root_dataset : Optional[Union[str, None]] = None

    
    
def store_base_dataset(cs: ConfigStore):
    cs.store(group="dataset", name="base_dataset_name", node=BaseDatasetConfig)