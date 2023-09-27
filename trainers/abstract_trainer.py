import math

import torch
import wandb

from log_utils import draw_samples, log
from networks import network_getter
from sampler import (sample_langevin_posterior, sample_langevin_prior,
                     sample_p_data)
from loss_reconstruction import get_loss_reconstruction


class AbstractTrainer():
    def __init__(self, cfg,) -> None:
        self.G, self.Encoder, self.E = network_getter(cfg["dataset"], cfg)
        self.cfg = cfg
        self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg["lr_G"], betas=(cfg["beta1_G"], cfg["beta2_G"]))
        self.optE = torch.optim.Adam(self.E.parameters(), lr=cfg["lr_E"], betas=(cfg["beta1_E"], cfg["beta2_E"]))
        self.optEncoder = torch.optim.Adam(self.Encoder.parameters(), lr=cfg["lr_Encoder"], betas=(cfg["beta1_Encoder"], cfg["beta2_Encoder"]))
        self.proposal = torch.distributions.normal.Normal(torch.tensor(cfg["proposal_mean"],device=cfg['device'], dtype=torch.float32),torch.tensor(cfg["proposal_std"],device=cfg['device'], dtype=torch.float32))
        self.base_dist = torch.distributions.normal.Normal(torch.tensor(0,device=cfg['device'], dtype=torch.float32),torch.tensor(1,device=cfg['device'], dtype=torch.float32))
        self.log_var_p = torch.tensor(0,device=cfg['device'], dtype=torch.float32)
        self.loss_reconstruction = get_loss_reconstruction(cfg)
        self.logger = wandb.init(project="LatentEBM", config=cfg)
        self.n_iter = cfg["n_iter"]
        self.n_iter_pretrain = cfg["n_iter_pretrain"]
        

    def train(self, train_data, val_data=None):
        i = 0
        for _ in range(self.n_iter_pretrain+self.n_iter):
            x = sample_p_data(train_data, self.cfg["batch_size"])
            if i < self.n_iter_pretrain:
                dic_loss = self.train_step_standard_elbo(x, i)
            else :
                dic_loss = self.train_step(x, i)
            if i % self.cfg["log_every"] == 0:
                log(i, dic_loss, logger=self.logger)
            if i % self.cfg["draw_every"] == 0:
                z_e_0, z_g_0 = self.base_dist.sample(64), self.base_dist.sample(64)
                z_e_k = sample_langevin_prior(z_e_0, self.E, self.cfg["K_0"], self.cfg["a_0"])
                z_g_k = sample_langevin_posterior(z_g_0, x, self.G, self.E, self.cfg["K_1"], self.cfg["a_1"], self.loss_reconstruction)
                x_base = self.G(z_e_0)
                x_prior = self.G(z_e_k)
                x_posterior = self.G(z_g_k)
                draw_samples(x_base, x_prior, x_posterior, None, self.cfg, i,  self.logger,)
            if i% self.cfg["val_every"] == 0 and val_data is not None:
                self.save(i)
             

    def eval(self, val_data, step, name="val/"):
        with torch.no_grad():
            z_e_0 = self.base_dist.sample(self.cfg["nb_sample_partition_estimate_val"])
            energy_base_dist = self.E(z_e_0).flatten(1).sum(1)
            base_dist_z_base_dist = self.base_dist.log_prob(z_e_0.flatten(1)).sum(1)
            log_partition_estimate = torch.logsumexp(-energy_base_dist -base_dist_z_base_dist,0) - math.log(energy_base_dist.shape[0])
            log(step, {"log_z":log_partition_estimate.item()}, logger=self.logger, name=name)
            k=0
            while k<len(val_data):
                x = val_data[k*self.cfg["batch_size"]:(k+1)*self.cfg["batch_size"]]
                dic = {}
                mu_q, log_var_q = self.Encoder().chunk(2,1)
                std_q = torch.exp(0.5*log_var_q)

                # Reparam trick
                eps = torch.randn_like(mu_q)
                z_q = (eps.mul(std_q).add_(mu_q)).reshape(-1,self.cfg["nz"],1,1)
                x_hat = self.G(z_q)


                # Reconstruction loss :
                loss_g = self.loss_reconstruction(x_hat, x)

                # KL without ebm
                KL_loss = 0.5 * (self.log_var_p - log_var_q -1 +  (log_var_q.exp() + mu_q.pow(2))/self.log_var_p.exp())
                KL_loss = KL_loss.sum(dim=1)

                # Entropy posterior
                entropy_posterior = torch.sum(0.5* (math.log(2*math.pi) +  log_var_q + 1), dim=1)

                # Energy :
                energy_approximate = self.E(z_q).flatten(1).sum(1)
                base_dist_z_approximate = self.base_dist.log_prob(z_q.flatten(1)).sum(1)


                loss_ebm = energy_approximate + log_partition_estimate.exp() -1
                loss_total = loss_g + KL_loss + loss_ebm

                dic_loss = {
                    "loss_g":loss_g,
                    "entropy_posterior":entropy_posterior,
                    "loss_ebm": loss_ebm,
                    "base_dist_z_approximate": base_dist_z_approximate,
                    "base_dist_z_base_dist" : base_dist_z_base_dist,
                    "KL_loss_no_ebm": KL_loss,
                    "energy_approximate": energy_approximate,
                    "energy_base_dist": energy_base_dist,
                    "approx_elbo" : -loss_total,
                }
                for key,value in dic_loss:
                    if key not in dic :
                        dic[key] = []
                    dic[key].append(value)
                k=k+1
            for key in dic:
                dic[key] = torch.stack(dic[key]).mean().item()
            log(step, dic, logger=self.logger, name=name)
            
            
    
    def test(self, data):
        raise NotImplementedError


    

    def train_step(self, x, step):
        raise NotImplementedError

    def train_step_standard_elbo(self, x, step):
        log_var_p = None
        self.optG.zero_grad()
        self.optE.zero_grad()
        self.optEncoder.zero_grad()

        z_e_0, z_g_0 = self.base_dist.sample(), self.base_dist.sample()
        mu_q, log_var_q = self.Encoder(x).chunk(2,1)

        # Reparametrization trick
        std_q = torch.exp(0.5*log_var_q)
        eps = torch.randn_like(mu_q)
        z_q = (eps.mul(std_q).add_(mu_q)).reshape(-1,self.cfg["nz"],1,1)

        # Reconstruction loss
        x_hat = self.G(z_q)
        loss_g = self.loss_reconstruction(x_hat, x)

        # KL loss
        KL_loss = 0.5 * (log_var_p - log_var_q -1 +  (log_var_q.exp() + mu_q.pow(2))/log_var_p.exp())
        KL_loss = KL_loss.sum(dim=1).mean(dim=0)

        # ELBO
        loss_total = loss_g + KL_loss
        loss_total.backward()

        dic_loss = {
            "loss_g":loss_g.item(),
            "KL_loss":KL_loss.item(),
            "elbo": -loss_total.item(),
        }
        self.optE.step()
        self.optG.step()
        self.optEncoder.step()

        return dic_loss
