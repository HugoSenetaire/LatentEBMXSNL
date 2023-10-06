import math

import torch

from ..Regularization.regularizer_ebm import regularization
from Model.Sampler.sampler_previous import (sample_langevin_posterior, sample_langevin_prior,
                     sample_p_0, Sampler)

from .abstract_trainer import AbstractTrainer


class Trainer_GaussianMixture(AbstractTrainer):
    def __init__(self, cfg, ):
        super().__init__(cfg, )

    def train_step(self, x, step):
        self.opt_generator.zero_grad()
        self.opt_energy.zero_grad()
        self.opt_encoder.zero_grad()
        self.opt_prior.zero_grad()

        z_e_0, z_g_0 = self.base_dist.sample((self.cfg.dataset.batch_size,self.cfg.trainer.nz,1,1)), self.base_dist.sample((self.cfg.dataset.batch_size,self.cfg.trainer.nz,1,1))
        mu_q, log_var_q = self.encoder(x).chunk(2,1)
        std_q = torch.exp(0.5*log_var_q)

        # Reparam trick
        eps = torch.randn_like(mu_q)
        z_q = (eps.mul(std_q).add_(mu_q)).reshape(-1,self.cfg.trainer.nz,1,1)
        x_hat = self.generator(z_q)


        # Reconstruction loss :
        loss_g = self.generator.get_loss(x_hat, x).mean(dim=0)

        # KL without ebm
        KL_loss = 0.5 * (self.log_var_p - log_var_q -1 +  (log_var_q.exp() + mu_q.pow(2))/self.log_var_p.exp())
        KL_loss = KL_loss.sum(dim=1).mean(dim=0)

        # Entropy posterior
        entropy_posterior = torch.sum(0.5* (math.log(2*math.pi) +  log_var_q + 1), dim=1).mean()

        # Energy :
        log_prob_multi_gaussian = self.prior.log_prob(z_q)
        loss_multi_gaussian = -log_prob_multi_gaussian.mean()
        base_dist_z_approximate = self.base_dist.log_prob(z_q.flatten(1)).sum(1)
        base_dist_z_base_dist = self.base_dist.log_prob(z_e_0.flatten(1)).sum(1)

        loss_total = loss_g - entropy_posterior + loss_multi_gaussian
        loss_total.backward()
        self.grad_clipping_all_net(self, ["energy", "generator", "encoder"], step)

        dic_loss = {
            "loss_g":loss_g.item(),
            "entropy_posterior":entropy_posterior.item(),
            "base_dist_z_approximate": base_dist_z_approximate.mean().item(),
            "base_dist_z_base_dist" : base_dist_z_base_dist.mean().item(),
            "KL_loss_no_ebm": KL_loss.item(),
            "loss_multi_gaussian": loss_multi_gaussian.mean().item(),
            "log_prob_multi_gaussian": log_prob_multi_gaussian.mean().item(),
            "elbo_no_ebm" : -loss_g.item() - KL_loss.item(),
            "elbo_mixture" : -loss_g.item() + entropy_posterior.item() - loss_multi_gaussian.item(),
            "mu_q": mu_q.flatten(1).mean(1).mean().item(),
            "log_var_q": log_var_q.flatten(1).sum(1).mean().item(),
        }


        self.opt_energy.step()
        self.opt_prior.step()
        if not self.cfg.trainer.fix_generator :
            self.opt_generator.step()
        if not self.cfg.trainer.fix_encoder :
            self.opt_encoder.step()
        return dic_loss