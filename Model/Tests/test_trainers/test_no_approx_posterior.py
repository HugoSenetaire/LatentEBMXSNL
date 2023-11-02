import sys
import os
import torch
import math
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
current_path.split('L')[0]
sys.path.append(current_path.split('Model')[0])


from Model.Tests.utils_test import get_config
from Model.Trainers import get_trainer
from Dataset.datasets import get_dataset_and_loader

def test_no_approx_reverse_posterior_supplement():

    cfg = get_config(config_name="conf")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.trainer.device = device

    train_dataloader, val_dataloader, test_dataloader, = get_dataset_and_loader(cfg, cfg.trainer.device)
    trainer = get_trainer(cfg)(cfg)
    mean_posterior = torch.tensor([1.0, 1.0])
    std_posterior = torch.tensor([3.0, 3.0])

    gaussian_posterior = torch.distributions.Normal(mean_posterior, std_posterior)

    iterator = iter(train_dataloader)
    for step in range(1000):
        try :
            x = next(iterator)[0].to(cfg.trainer.device)
        except StopIteration:
            iterator = iter(train_dataloader)
            x = next(iterator)[0].to(cfg.trainer.device)
        samples = gaussian_posterior.sample((x.shape[0],)).to(cfg.trainer.device)
        trainer.train_approx_posterior_reverse(x =x , z_g_k = samples, step = step)

    try :
        x = next(iterator)[0].to(cfg.trainer.device)
    except StopIteration:
        iterator = iter(train_dataloader)
        x = next(iterator)[0].to(cfg.trainer.device)

    params_posterior = trainer.reverse_encoder(x)
    mu, log_var = params_posterior.chunk(2, 1)
    assert torch.allclose(mu.mean(0), mean_posterior, rtol=1), "The mean should be the same"
    assert torch.allclose(log_var.mean(0), 2*torch.log(std_posterior), rtol=1), "The log var should be the same, log_var_calculated {}, init {}".format(log_var.mean(0), 2*torch.log(std_posterior))
    assert trainer.reverse_encoder.latent_distribution

