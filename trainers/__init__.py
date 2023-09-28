from .abstract_trainer import AbstractTrainer
from .train_cd import TrainerCD
# from .train_cd_trick import TrainerCDTrick
from .train_lebm_snl import Trainer_LEBM_SNL
from .train_lebm_snlv2 import Trainer_LEBM_SNL2

def get_trainer(cfg):
    trainer_name = cfg["trainer"]
    if trainer_name == "TrainerCD":
        return TrainerCD
    # elif trainer_name == "TrainerCDTrick":
        # return TrainerCDTrick
    elif trainer_name == "Trainer_LEBM_SNL":
        return Trainer_LEBM_SNL
    elif trainer_name == "Trainer_LEBM_SNL2":
        return Trainer_LEBM_SNL2
    else:
        raise ValueError("Trainer not implemented")