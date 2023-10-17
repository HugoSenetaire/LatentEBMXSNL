import math

import torch

from ..Regularization import regularization_encoder

from .abstract_trainer import AbstractTrainer


class TrainerPrior(AbstractTrainer):
    def __init__(self, cfg, ):
        super().__init__(cfg, )
        self.detach_approximate_posterior = cfg.trainer.detach_approximate_posterior

    def train_step(self, x, step):
        dic_feedback = {}
        self.opt_generator.zero_grad()
        self.opt_energy.zero_grad()
        self.opt_encoder.zero_grad()
        self.opt_prior.zero_grad()

        z_e_0, z_g_0 = self.prior.sample(x.shape[0]), self.prior.sample(x.shape[0])
        param = self.encoder(x)
        dic_param, dic_param_feedback = self.encoder.latent_distribution.get_params(param)
        dic_feedback.update(dic_param_feedback)


        # Reparam trick
        z_q = self.encoder.latent_distribution.r_sample(param).reshape(x.shape[0], self.cfg.trainer.nz)
        x_hat = self.generator(z_q)


        # Reconstruction loss :
        loss_g = self.generator.get_loss(x_hat, x).reshape(x.shape[0]).mean(dim=0)

        # KL without ebm
        KL_loss = self.encoder.latent_distribution.calculate_kl(self.prior, param, z_q, dic_params=dic_param, empirical_kl=self.cfg.trainer.empirical_kl).mean(dim=0)

        # Entropy posterior
        entropy_posterior = self.encoder.latent_distribution.calculate_entropy(param, dic_params=dic_param, empirical_entropy=self.cfg.trainer.empirical_entropy).mean(dim=0)

        # Energy :
        if self.detach_approximate_posterior:
            z_q = z_q.detach()
        log_prob_multi_gaussian = self.extra_prior.log_prob(z_q).reshape(x.shape[0])
        loss_multi_gaussian = -log_prob_multi_gaussian.mean()


        base_dist_z_approximate = self.prior.log_prob(z_q).reshape(x.shape[0])
        base_dist_z_base_dist = self.prior.log_prob(z_e_0).reshape(x.shape[0])
        if self.detach_approximate_posterior:
            loss_total = loss_g + KL_loss + loss_multi_gaussian # Train the vae with gaussian prior and just focus the new prior
        else :
            loss_total = loss_g - entropy_posterior + loss_multi_gaussian # Train the var directly with other prior

        dic_loss_regul_enc = regularization_encoder(dic_param, self.encoder, self.cfg, self.logger, step)
        for key, item in dic_loss_regul_enc.items():
            loss_total += item
        dic_feedback.update(dic_loss_regul_enc)
        loss_total.backward()
        self.grad_clipping_all_net(["energy", "generator", "encoder"], step)

        dic_feedback.update({
            "loss_g":loss_g.item(),
            "entropy_posterior":entropy_posterior.item(),
            "base_dist_z_approximate": base_dist_z_approximate.mean().item(),
            "base_dist_z_base_dist" : base_dist_z_base_dist.mean().item(),
            "KL_loss_no_ebm": KL_loss.item(),
            "loss_multi_gaussian": loss_multi_gaussian.mean().item(),
            "log_prob_multi_gaussian": log_prob_multi_gaussian.mean().item(),
            "elbo_no_ebm" : -loss_g.item() - KL_loss.item(),
            "elbo_mixture" : -loss_g.item() + entropy_posterior.item() - loss_multi_gaussian.item(),
            })
        dic_feedback.update(dic_param)

        self.opt_energy.step()
        self.opt_prior.step()
        if not self.cfg.trainer.fix_generator :
            self.opt_generator.step()
        if not self.cfg.trainer.fix_encoder :
            self.opt_encoder.step()
        return dic_feedback