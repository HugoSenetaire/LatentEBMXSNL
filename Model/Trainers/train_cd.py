import math

from ..Regularization import regularization, regularization_encoder
from .abstract_trainer import AbstractTrainer
import torch




class ContrastiveDivergence(AbstractTrainer):
    def __init__(self, cfg, ):
        super().__init__(cfg, )
        
    def train_step(self, x, step):
        dic_total = {}
        z_e_0, z_g_0 = self.prior.sample(x.shape[0]), self.prior.sample(x.shape[0])
        z_e_k, z_grad_norm = self.sampler_prior(z_e_0, self.energy, self.prior,)
        z_g_k, z_g_grad_norm, z_e_grad_norm = self.sampler_posterior(z_g_0, x, self.generator, self.energy, self.prior,)

        z_e_0_norm = z_e_0.detach().norm(dim=1).mean()
        z_e_k_norm = z_e_k.detach().norm(dim=1).mean()
        z_g_k_norm = z_g_k.detach().norm(dim=1).mean()

        dic_total.update({
            "z_e_0_norm": z_e_0_norm,
            "z_e_k_norm": z_e_k_norm,
            "z_g_k_norm": z_g_k_norm,
            "z_e_k_grad_norm_e": z_grad_norm,
            "z_g_k_grad_norm_g": z_g_grad_norm,
            "z_g_k_grad_norm_e": z_e_grad_norm,
            "lr_e": self.opt_energy.param_groups[0]["lr"],
            "lr_g": self.opt_generator.param_groups[0]["lr"],
        })


        self.opt_encoder.zero_grad()
        self.opt_generator.zero_grad()
        x_hat = self.generator(z_g_k.detach())
        loss_g = self.generator.get_loss(x_hat, x).mean(dim=0).mean()
        mse_loss = self.mse(x_hat, x) / x.shape[0]
        # mse_loss.backward()
        loss_g.backward()
        self.grad_clipping_all_net(["generator"], step=step)
        self.opt_generator.step()


        self.opt_energy.zero_grad()
        en_pos, en_neg = self.energy(z_g_k.detach()).mean(), self.energy(z_e_k.detach()).mean()
        loss_e = en_pos - en_neg
        dic_regul = regularization(self.energy, z_g_k, z_e_k, en_pos, en_neg, self.cfg, self.logger, step)
        for key, item in dic_regul.items():
            loss_e += item
        dic_total.update(dic_regul)
        loss_e.backward()
        self.grad_clipping_all_net(["energy"], step=step)
        self.opt_energy.step()



        # Train the encoder to go to regular ebm, not really the same thing, it's just so that I get better approximation, could do reverse KL ?
        self.opt_encoder.zero_grad()
        params = self.encoder(x)
        dic_params, dic_params_feedback = self.encoder.latent_distribution.get_params(params)
        dic_total.update(dic_params_feedback)
        
        # Reparam trick
        z_q = self.encoder.latent_distribution.r_sample(params, dic_params=dic_params).reshape(x.shape[0], self.cfg.trainer.nz)
        x_hat = self.generator(z_q)
        
        # Reconstruction loss :
        loss_g = self.generator.get_loss(x_hat, x).reshape(x.shape[0]).mean(dim=0)

        # KL without ebm
        KL_loss = self.encoder.latent_distribution.calculate_kl(self.prior, params, z_q, dic_params=dic_params).mean(dim=0)


        # Entropy posterior
        entropy_posterior = self.encoder.latent_distribution.calculate_entropy(params, dic_params=dic_params).mean(dim=0)

        loss_elbo = loss_g + KL_loss
        dic_regul_encoder = regularization_encoder(dic_params, self.encoder, self.cfg, self.logger, step)
        for key, item in dic_regul_encoder.items():
            loss_elbo += item
        loss_elbo.backward()
        self.grad_clipping_all_net(["encoder"], step=step)
        self.opt_encoder.step()



        

        dic_total.update({
            "loss_e": loss_e,
            "loss_g": loss_g,
            "mse_loss": mse_loss,
            "en_pos": en_pos,
            "en_neg": en_neg,
            "loss_elbo": loss_elbo,
            "KL_loss": KL_loss,
            "entropy_posterior": entropy_posterior,
            })
        
        
        return dic_total