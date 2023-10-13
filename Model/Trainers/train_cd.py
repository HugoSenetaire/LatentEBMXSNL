import math

from ..Regularization.regularizer_ebm import regularization
from .abstract_trainer import AbstractTrainer
import torch




class ContrastiveDivergence(AbstractTrainer):
    def __init__(self, cfg, ):
        super().__init__(cfg, )
        
    def train_step(self, x, step):
        z_e_0, z_g_0 = self.base_dist.sample(x.shape[0]), self.base_dist.sample(x.shape[0])
        z_e_k, z_grad_norm = self.sampler_prior(z_e_0, self.energy, self.base_dist,)
        z_g_k, z_g_grad_norm, z_e_grad_norm = self.sampler_posterior(z_g_0, x, self.generator, self.energy, self.base_dist,)

        z_e_0_norm = z_e_0.detach().norm(dim=1).mean()
        z_e_k_norm = z_e_k.detach().norm(dim=1).mean()
        z_g_k_norm = z_g_k.detach().norm(dim=1).mean()

        dic_loss={
            "z_e_0_norm": z_e_0_norm,
            "z_e_k_norm": z_e_k_norm,
            "z_g_k_norm": z_g_k_norm,
            "z_e_k_grad_norm_e": z_grad_norm,
            "z_g_k_grad_norm_g": z_g_grad_norm,
            "z_g_k_grad_norm_e": z_e_grad_norm,
            "lr_e": self.opt_energy.param_groups[0]["lr"],
            "lr_g": self.opt_generator.param_groups[0]["lr"],
        }


        self.opt_encoder.zero_grad()
        self.opt_generator.zero_grad()
        x_hat = self.generator(z_g_k.detach())
        loss_g = self.generator.get_loss(x_hat, x).mean(dim=0).mean()
        mse_loss = self.mse(x_hat, x) / x.shape[0]
        mse_loss.backward()
        self.opt_generator.step()


        self.opt_energy.zero_grad()
        en_pos, en_neg = self.energy(z_g_k.detach()).mean(), self.energy(z_e_k.detach()).mean()
        loss_e = en_pos - en_neg
        dic_loss = regularization(self.energy, z_g_k, z_e_k, en_pos, en_neg, self.cfg, self.logger, step)
        for key, item in dic_loss.items():
            loss_e += item
        loss_e.backward()
        self.opt_energy.step()



        # Train the encoder to go to regular ebm, not really the same thing, it's just so that I get better approximation, could do reverse KL ?
        self.opt_encoder.zero_grad()
        param = self.encoder(x)
        mu_q, log_var_q = param.chunk(2,1)
        std_q = torch.exp(0.5*log_var_q)
        # Reparam trick
        eps = torch.randn_like(mu_q)
        z_q = (eps.mul(std_q).add_(mu_q))
        x_hat = self.generator(z_q)
        # Reconstruction loss :
        loss_g = self.generator.get_loss(x_hat, x).reshape(x.shape[0]).mean(dim=0)

        # KL without ebm
        KL_loss = 0.5 * (self.log_var_p - log_var_q -1 +  (log_var_q.exp() + mu_q.pow(2))/self.log_var_p.exp())
        KL_loss = KL_loss.reshape(x.shape[0],self.cfg.trainer.nz).sum(dim=1).mean(dim=0)

        loss_elbo = loss_g + KL_loss
        loss_elbo.backward()
        self.opt_encoder.step()



        

        dic_loss.update({
            "loss_e": loss_e,
            "loss_g": loss_g,
            "mse_loss": mse_loss,
            "en_pos": en_pos,
            "en_neg": en_neg,
            })
        
        return dic_loss