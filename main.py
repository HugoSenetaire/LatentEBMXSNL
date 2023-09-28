import argparse
import pathlib

import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tfm

from datasets import get_dataset_and_loader
from trainers import get_trainer

trainer = "Trainer_LEBM_SNL"
dataset = "BINARYMNIST"
save_image_every = 50
log_every = 10
n_iter = 70000
n_iter_pretrain = 1000
val_every = 100
nb_sample_partition_estimate_val = 20000
multiple_sample_val = 10
root = "/scratch/hhjs/data"


if dataset == "SVHN_original":
    img_size, batch_size = 32, 256
    nz, nc, ndf, ngf = 100, 3, 200, 64
    K_0, a_0, K_1, a_1 = 60, 0.4, 40, 0.1
    llhd_sigma = 0.3
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    loss_reconstruction = "gaussian"

elif dataset == "SVHN":
    img_size, batch_size = 32, 256
    nz, nc, ndf, ngf = 100, 3, 200, 64
    K_0, a_0, K_1, a_1 = 20, 0.4, 20, 0.1
    llhd_sigma = 0.3
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    loss_reconstruction = "gaussian"


elif dataset == "MNIST":
    img_size, batch_size = 28, 256
    nz, nc, ndf, ngf = 16, 1, 200, 16
    K_0, a_0, K_1, a_1 = 20, 0.4, 20, 0.1
    llhd_sigma = 0.3
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    loss_reconstruction = "gaussian"

elif dataset == "BINARYMNIST":
    img_size, batch_size = 28, 256
    nz, nc, ndf, ngf = 16, 1, 200, 16
    K_0, a_0, K_1, a_1 = 20, 0.4, 20, 0.1
    llhd_sigma = 0.3
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    loss_reconstruction = "bernoulli"

elif dataset == "CIFAR_10":
    img_size, batch_size = 32, 256
    nz, nc, ndf, ngf = 16, 1, 200, 16
    K_0, a_0, K_1, a_1 = 20, 0.4, 20, 0.1
    llhd_sigma = 0.3
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    loss_reconstruction = "gaussian"


cfg = {
    "root": root,
    "trainer": trainer,
    "save_image_every": save_image_every,
    "val_every": val_every,
    "multiple_sample_val": multiple_sample_val,
    "nb_sample_partition_estimate_val": nb_sample_partition_estimate_val,
    "n_iter_pretrain": n_iter_pretrain,
    "log_every": log_every,
    "dataset": dataset,
    "img_size": img_size,
    "batch_size": batch_size,
    "nz": nz,
    "nc": nc,
    "ndf": ndf,
    "ngf": ngf,
    "K_0": K_0,
    "a_0": a_0,
    "K_1": K_1,
    "a_1": a_1,
    "llhd_sigma": llhd_sigma,
    "n_iter": n_iter,
    "device": device,
    "loss_reconstruction": loss_reconstruction,
}


cfg_clipping = {
    "E_clip_grad_type": 'norm',
    "E_clip_grad_value": 1.0,
    "E_nb_sigma": 3,
    "E_replace_nan": False,
    "G_clip_grad_type": 'norm',
    "G_clip_grad_value": 1.0,
    "G_nb_sigma": 3,
    "G_replace_nan": False,
    "Encoder_clip_grad_type": 'norm',
    "Encoder_clip_grad_value": 1.0,
    "Encoder_nb_sigma": 3,
    "Encoder_replace_nan": False,
}

cfg_regularization = {
    "l2_grad": 0.0,
    "l2_param": 0.0,
    "l2_output": 0.0,
}

cfg_proposal = {
    "proposal_mean": 0.0,
    "proposal_std": 1.0,
}

cfg.update(cfg_proposal)
cfg.update(cfg_clipping)
cfg.update(cfg_regularization)

cfg.update(
    {
        "lr_E": 0.00005,
        "beta1_E": 0.5,
        "beta2_E": 0.999,
        "lr_G": 0.0001,
        "beta1_G": 0.5,
        "beta2_G": 0.999,
        "lr_Encoder": 0.0001,
        "beta1_Encoder": 0.5,
        "beta2_Encoder": 0.999,
        "fix_encoder": False,
        "fix_decoder": False,
    }
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script with argparse")
    parser.add_argument("-k", "--karolina", help="bool", action="store_true")
    args = parser.parse_args()

    if args.karolina:
        dir_name = pathlib.Path(
            pathlib.Path.home().parent.parent, "scratch/project/dd-23-57/"
        )
        cfg.update({"root": dir_name})
        log_dir = pathlib.Path(
            pathlib.Path.home().parent.parent, "scratch/project/dd-23-57/wandb_log/LEBM"
        )
        cfg.update({"log_dir": log_dir})
    else:
        cfg.update({"root": "/scratch/hhjs/data"})
        cfg.update({"log_dir": "/scratch/hhjs/logs"})
    data_train, data_valid, data_test, transform_back_name = get_dataset_and_loader(cfg, device)
    cfg.update({"transform_back_name": transform_back_name})

    total_train = get_trainer(cfg)
    total_train = total_train(cfg)
    total_train.train(train_data=data_train, val_data=data_test)
