from .abstract_trainer import AbstractTrainer
from .train_cd import ContrastiveDivergence
from .train_cd_trick import ContrastiveDivergenceLogTrick
from .train_lebm_snl import SNELBO
from .train_prior import TrainerPrior
from .train_lebm_snl_aggregateposterior import SNELBO_Aggregate

def get_trainer(cfg):
    trainer_name = cfg.trainer.trainer_name
    if trainer_name == "contrastive_divergence_trainer":
        return ContrastiveDivergence
    elif trainer_name == "contrastive_divergence_log_trick_trainer":
        return ContrastiveDivergenceLogTrick
    elif trainer_name == "snelbo_trainer":
        return SNELBO
    elif trainer_name == "snelbo_aggregate_trainer":
        return SNELBO_Aggregate
    elif trainer_name == "prior_trainer":
        return TrainerPrior
    else:
        raise ValueError("Trainer not implemented")