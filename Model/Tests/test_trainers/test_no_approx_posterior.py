import sys
import os
import torch
import math
import tqdm
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
current_path.split('L')[0]
sys.path.append(current_path.split('Model')[0])

from Model.Tests.utils_test import get_config
from Model.Trainers import get_trainer
from Dataset.datasets import get_dataset_and_loader

def test_no_approx_reverse_posterior():

    cfg = get_config(config_name="conf")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.trainer.device = device

    train_dataloader, val_dataloader, test_dataloader, = get_dataset_and_loader(cfg, cfg.trainer.device)
    trainer = get_trainer(cfg)(cfg)
    mean_posterior = torch.tensor([1.0, 1.0]).to(cfg.trainer.device)
    std_posterior = torch.tensor([3.0, 3.0]).to(cfg.trainer.device)

    gaussian_posterior = torch.distributions.Normal(mean_posterior, std_posterior)

    iterator = iter(train_dataloader)
    for step in tqdm.tqdm(range(1000)):
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



def test_no_approx_reverse_posterior_supplement():

    cfg = get_config(config_name="conf")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.trainer.device = device

    trainer = get_trainer(cfg)(cfg)
    mean_1_posterior = torch.tensor([10.0, 10.0]).to(cfg.trainer.device)
    mean_2_posterior = torch.tensor([-10.0, -10.0]).to(cfg.trainer.device)
    std_posterior = torch.tensor([3.0, 3.0]).to(cfg.trainer.device)

    # gaussian_posterior = torch.distributions.Normal(mean_posterior, std_posterior)

    for step in tqdm.tqdm(range(10000)):
        x1 = -torch.ones((int(cfg.dataset.batch_size/2), cfg.dataset.nc, cfg.dataset.img_size, cfg.dataset.img_size)).to(cfg.trainer.device) + 0.5 * torch.randn((int(cfg.dataset.batch_size/2), cfg.dataset.nc, cfg.dataset.img_size, cfg.dataset.img_size)).to(cfg.trainer.device)
        x2 = torch.ones((int(cfg.dataset.batch_size/2), cfg.dataset.nc, cfg.dataset.img_size, cfg.dataset.img_size)).to(cfg.trainer.device) +  0.5 * torch.randn((int(cfg.dataset.batch_size/2), cfg.dataset.nc, cfg.dataset.img_size, cfg.dataset.img_size)).to(cfg.trainer.device)
        x = torch.cat((x1, x2), dim=0)
        # samples = gaussian_posterior.sample((x.shape[0],)).to(cfg.trainer.device)
        posterior_1 = torch.distributions.Normal(mean_1_posterior, std_posterior)
        posterior_2 = torch.distributions.Normal(mean_2_posterior, std_posterior)
        samples = torch.cat((posterior_1.sample((int(cfg.dataset.batch_size/2),)), posterior_2.sample((int(cfg.dataset.batch_size/2),))), dim=0).to(cfg.trainer.device)
        trainer.train_approx_posterior_reverse(x=x , z_g_k = samples, step = step)


    x = -torch.ones((int(cfg.dataset.batch_size/2), cfg.dataset.nc, cfg.dataset.img_size, cfg.dataset.img_size)).to(cfg.trainer.device) + 0.5 * torch.randn((int(cfg.dataset.batch_size/2), cfg.dataset.nc, cfg.dataset.img_size, cfg.dataset.img_size)).to(cfg.trainer.device)
    param_1_posterior = trainer.reverse_encoder(x)
    mu_1, log_var = param_1_posterior.chunk(2, 1)
    print(mu_1)
    print(mean_1_posterior)
    assert torch.allclose(mu_1.mean(0), mean_1_posterior, rtol=1e-1), "The mean should be the same, mean_calculated {}, init {}".format(mu_1.mean(0), mean_1_posterior)
    assert torch.allclose(log_var.mean(0), 2*torch.log(std_posterior), rtol=1e-1), "The log var should be the same, log_var_calculated {}, init {}".format(log_var.mean(0), 2*torch.log(std_posterior))

    x = torch.ones((int(cfg.dataset.batch_size/2), cfg.dataset.nc, cfg.dataset.img_size, cfg.dataset.img_size)).to(cfg.trainer.device) + 0.5 * torch.randn((int(cfg.dataset.batch_size/2), cfg.dataset.nc, cfg.dataset.img_size, cfg.dataset.img_size)).to(cfg.trainer.device)
    param_2_posterior = trainer.reverse_encoder(x)
    mu_2, log_var = param_2_posterior.chunk(2, 1)
    print(mu_2)
    print(mean_2_posterior)
    assert torch.allclose(mu_2.mean(0), mean_2_posterior, rtol=1e-1), "The mean should be the same, mean_calculated {}, init {}".format(mu_2.mean(0), mean_2_posterior)
    assert torch.allclose(log_var.mean(0), 2*torch.log(std_posterior), rtol=1e-1), "The log var should be the same, log_var_calculated {}, init {}".format(log_var.mean(0), 2*torch.log(std_posterior))

    x = torch.zeros((int(cfg.dataset.batch_size/2), cfg.dataset.nc, cfg.dataset.img_size, cfg.dataset.img_size)).to(cfg.trainer.device) + 0.5 * torch.randn((int(cfg.dataset.batch_size/2), cfg.dataset.nc, cfg.dataset.img_size, cfg.dataset.img_size)).to(cfg.trainer.device)
    param_3_posterior = trainer.reverse_encoder(x)
    mu_3, log_var = param_3_posterior.chunk(2, 1)
    print(mu_3)

if __name__ == "__main__":
    # test_no_approx_reverse_posterior()
    test_no_approx_reverse_posterior_supplement()