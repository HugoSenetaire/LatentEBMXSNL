import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
from omegaconf import DictConfig

@dataclass
class BaseTrainerConfig:
    trainer_name : str = MISSING
    log_dir : Optional[Union[str, None]] = None

    # Latent space :
    nz : int = MISSING

    # Used by the trainer to estimate the normalization constant
    proposal_mean: Optional[float] = 0.0
    proposal_std: Optional[float] = 1.0

    # Iteration for training
    n_iter: int = 100000
    n_iter_pretrain: int = 0

    #Save utils
    log_every: int = 100
    save_every: int = 1000
    save_images_every: int = 1000
    val_every: int = 1000
    test_every: int = 10000

    # Validation parameters :
    multiple_sample_val_SNIS: int = 100
    multiple_sample_val: int = 100
    nb_sample_partition_estimate_val: int = 100

    device : str = "cuda"


@dataclass
class TrainerContrastiveDivergence(BaseTrainerConfig):
    trainer_name : str = "contrastive_divergence_trainer"
    use_trick_sampler : bool = False

@dataclass
class TrainerContrastiveDivergenceLogTrick(BaseTrainerConfig):
    trainer_name : str = "contrastive_divergence_log_trick_trainer"
    use_trick_sampler : bool = False

@dataclass    
class TrainerPrior(BaseTrainerConfig):
    trainer_name : str = "prior_trainer"
    detach_approximate_posterior : bool = False
    fix_encoder : bool = False
    fix_generator : bool = False

        

@dataclass
class TrainerSNELBO(BaseTrainerConfig):
    trainer_name : str = "snelbo_trainer"
    detach_approximate_posterior : bool = False
    fix_encoder : bool = False
    fix_generator : bool = False

def store_base_trainer(cs: ConfigStore):
    cs.store(group="trainer", name="contrastive_divergence_trainer_base", node=TrainerContrastiveDivergence)
    cs.store(group="trainer", name="contrastive_divergence_log_trick_trainer_base", node=TrainerContrastiveDivergenceLogTrick)
    cs.store(group="trainer", name="prior_trainer_base", node=TrainerPrior)
    cs.store(group="trainer", name="snelbo_trainer_base", node=TrainerSNELBO)


    
    