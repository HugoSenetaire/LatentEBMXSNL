import logging
import os
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from Model.Encoder.hydra_config import store_base_encoder, BaseEncoderConfig
from Model.Energy.hydra_config import store_base_energy, BaseEnergyConfig
from Model.Generator.hydra_config import store_base_generator, BaseGeneratorConfig
from Model.Optim.hydra_config import store_base_optim, BaseOptimConfig
from Model.Sampler.hydra_config import store_base_langevin_sampler, BaseLangevinSampler
from Model.Trainers.hydra_config import store_base_trainer, BaseTrainerConfig
from Model.Prior.hydra_config import store_base_prior, BasePriorConfig
from Dataset.hydra_config import store_base_dataset, BaseDatasetConfig
from Model.Regularization.hydra_config import store_base_regularization, BaseRegularizationConfig

@dataclass
class MachineConfig:
    root: str = MISSING

@dataclass 
class KarolinaConfig(MachineConfig):
    root: str = MISSING

@dataclass
class TitansConfig(MachineConfig):
    root: str = "/scratch/hhjs/"

@dataclass
class Config:
    dataset: BaseDatasetConfig = MISSING
    encoder: BaseEncoderConfig = MISSING
    energy: BaseEnergyConfig = MISSING
    generator: BaseGeneratorConfig = MISSING
    prior : BasePriorConfig = MISSING
    regularization: BaseRegularizationConfig = MISSING
    sampler_prior: BaseLangevinSampler = MISSING
    sampler_posterior: BaseLangevinSampler = MISSING
    sampler_prior_no_trick: BaseLangevinSampler = MISSING
    sampler_posterior_no_trick: BaseLangevinSampler = MISSING
    optim_encoder: BaseOptimConfig = MISSING
    optim_energy: BaseOptimConfig = MISSING
    optim_generator: BaseOptimConfig = MISSING
    optim_prior: BaseOptimConfig = MISSING
    trainer: BaseTrainerConfig = MISSING
    machine: MachineConfig = MISSING

    def __post_init__(self,):
        # assert False
        if self.trainer.log_dir is None:
            print("Setting log dir to " + os.path.join(self.machine.root, "log"))
            self.trainer.log_dir = os.path.join(self.machine.root, "log")
        else :
            print("Using log dir " + self.trainer.log_dir)
        if self.dataset.root_dataset is None:
            print("Setting root dataset to " + os.path.join(self.machine.root, "data"))
            self.dataset.root_dataset = os.path.join(self.machine.root, "data")
        else :
            print("Using root dataset " + self.dataset.root_dataset)


def store_main():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)

    cs.store(name="machine", node=MachineConfig, group="machine")
    cs.store(name="karolina", node=KarolinaConfig, group="machine")
    cs.store(name="titans", node=TitansConfig, group="machine")

    store_base_encoder(cs)
    store_base_energy(cs)
    store_base_generator(cs)
    store_base_prior(cs)

    store_base_optim(cs)

    store_base_langevin_sampler(cs)

    store_base_trainer(cs)

    store_base_dataset(cs)

    store_base_regularization(cs)


@hydra.main(version_base="1.1", config_name="conf", config_path="conf_mnist")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    store_main()
    main()