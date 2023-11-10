import sys
import os
import torch
import math
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path.split('Model')[0])
from Model.Tests.utils_test import get_config

from Model.Encoder import AbstractEncoder
from Model.Prior import get_prior

from Model.Encoder.latent_distribution.uniform import UniformPosterior
from Model.Prior.hyperspherical_uniform_prior import HypersphericalUniform
from Model.Generator import AbstractGenerator
from Model.Trainers import get_trainer
from Dataset.datasets import get_dataset_and_loader
import matplotlib.pyplot as plt



def test_kl_von_mises_fischer():
    cfg = get_config(config_path="conf_mnist_2dlatent", config_name="conf_hyperspherical")
    cfg.prior.prior_name = 'hyperspherical_uniform'
    cfg.encoder.latent_distribution_name = 'von_mises_fischer'
    cfg.trainer.device = "cpu"
    trainer = get_trainer(cfg)(cfg)
    encoder = trainer.encoder


    data_train, data_val, data_test = get_dataset_and_loader(cfg, cfg.trainer.device)

    iterat = iter(data_train)
    batch = next(iterat)
    data = batch[0].to(cfg.trainer.device)
    targets = batch[1].to(cfg.trainer.device)
    while len(data)<5000:
        batch = next(iterat)
        data = torch.cat([data,batch[0].to(cfg.trainer.device)], dim=0)
        targets = torch.cat([targets, batch[1].to(cfg.trainer.device)], dim=0)
    data = data[:5000]
    targets = targets[:5000]
    len_samples = min(5000, data.shape[0])

    param = encoder(data)
    dir_save = os.path.join(current_path, "test")
    dic_params, _ = encoder.latent_distribution.get_params(param)
    assert torch.allclose(dic_params['z_mean'].norm(dim=-1, keepdim=True), torch.ones_like(dic_params['z_mean'])), "The norm should be 1"

    encoder.init_network(data_train)
    param_2 = encoder(data)
    dic_params_2, _ = encoder.latent_distribution.get_params(param_2)
    assert torch.allclose(dic_params_2['z_mean'].norm(dim=-1, keepdim=True), torch.ones_like(dic_params_2['z_mean'])), "The norm should be 1"

    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    plt.scatter(dic_params['z_mean'].detach().cpu().numpy()[:,0], dic_params['z_mean'].detach().cpu().numpy()[:,1], color = 'red', alpha = 0.5)
    plt.scatter(param.detach().cpu().numpy()[:,0], param.detach().cpu().numpy()[:,1], color = 'blue', alpha =0.5)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)    
    plt.savefig(os.path.join(dir_save, "img_test_own_param.png"))
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.savefig(os.path.join(dir_save, "img_test_own_param_2.png"))
    plt.close()


    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    plt.scatter(dic_params_2['z_mean'].detach().cpu().numpy()[:,0], dic_params_2['z_mean'].detach().cpu().numpy()[:,1], color = 'red', alpha = 0.5)
    plt.scatter(param_2.detach().cpu().numpy()[:,0], param_2.detach().cpu().numpy()[:,1], color = 'blue', alpha =0.5)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)    
    plt.savefig(os.path.join(dir_save, "img_test_own_param_post.png"))
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.savefig(os.path.join(dir_save, "img_test_own_param_2_post.png"))
    plt.close()

    


   
if __name__ == "__main__":
    test_kl_von_mises_fischer()