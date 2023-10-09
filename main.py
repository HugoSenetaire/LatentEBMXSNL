import argparse
import pathlib
import torch

import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tfm

from Dataset.datasets import get_dataset_and_loader
from Model.Trainers import get_trainer

import hydra
from omegaconf import OmegaConf

import hydra_config


root = "/scratch/hhjs/data"


@hydra.main(version_base="1.1", config_path="conf_mnist", config_name="conf",)
def main(cfg):
    device = "cuda" if t.cuda.is_available() else "cpu"
    cfg.trainer.device = device

    data_train, data_valid, data_test = get_dataset_and_loader(cfg, device)
    if isinstance(data_valid, torch.utils.data.dataloader.DataLoader,):
        data_valid = data_valid.dataset.tensors[0]
    if isinstance(data_test, torch.utils.data.dataloader.DataLoader,):
        data_test = data_test.dataset.tensors[0]

    total_train = get_trainer(cfg)
    total_train = total_train(cfg)
    total_train.train(train_data=data_train, val_data=data_test)


if __name__ == "__main__":
    hydra_config.store_main()


    
        
    main()
