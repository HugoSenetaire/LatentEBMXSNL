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
    n_iter_pretrain_encoder: int = 0

    #Save utils
    log_every: int = 100
    save_every: int = 1000
    save_images_every: int = 1000
    val_every: int = 1000
    test_every: int = 10000

    # Validation parameters :
    multiple_sample_val_SNIS: int = 10
    multiple_sample_val: int = 10
    nb_sample_fid_val: int= 10000
    nb_sample_partition_estimate_val: int = 10

    # Test parameters :
    multiple_sample_test_SNIS: int = 1000
    multiple_sample_test: int = 1000
    nb_sample_fid_test: int= 30000
    nb_sample_partition_estimate_test: int = 1000

    # Train parameters :
    multiple_sample_train: int = 10

    # Plot parameters :
    grid_coarseness : int = 100

    # Calculate KL and Entropy with MC :
    empirical_kl : bool = False
    empirical_entropy : bool = False


    device : str = "cuda"


@dataclass
class TrainerContrastiveDivergence(BaseTrainerConfig):
    trainer_name : str = "contrastive_divergence_trainer"

@dataclass
class TrainerContrastiveDivergenceLogTrick(BaseTrainerConfig):
    trainer_name : str = "contrastive_divergence_log_trick_trainer"

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
    empirical_kl : bool = False
    empirical_entropy : bool = False

@dataclass
class TrainerSNELBOAggregatePosterior(BaseTrainerConfig):
    trainer_name : str = "snelbo_aggregate_trainer"
    detach_approximate_posterior : bool = False
    fix_encoder : bool = False
    fix_generator : bool = False

def store_base_trainer(cs: ConfigStore):
    cs.store(group="trainer", name="contrastive_divergence_trainer_base", node=TrainerContrastiveDivergence)
    cs.store(group="trainer", name="contrastive_divergence_log_trick_trainer_base", node=TrainerContrastiveDivergenceLogTrick)
    cs.store(group="trainer", name="prior_trainer_base", node=TrainerPrior)
    cs.store(group="trainer", name="snelbo_trainer_base", node=TrainerSNELBO)
    cs.store(group="trainer", name="snelbo_aggregate_trainer_base", node=TrainerSNELBOAggregatePosterior)


    
    