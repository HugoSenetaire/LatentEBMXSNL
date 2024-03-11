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


@hydra.main(version_base="1.1", config_path="conf_celeba", config_name="conf_snelbo",)
def main(cfg):
    device = "cuda:0" if t.cuda.is_available() else "cpu"
    cfg.trainer.device = device

    data_train, data_val, data_test = get_dataset_and_loader(cfg, device)

    total_train = get_trainer(cfg)
    total_train = total_train(cfg)
    total_train.train(train_dataloader=data_train, val_dataloader=data_val, test_dataloader=data_test)


if __name__ == "__main__":
    hydra_config.store_main()
    main()
