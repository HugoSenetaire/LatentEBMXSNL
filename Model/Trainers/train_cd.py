import math

from ..Regularization import regularization, regularization_encoder
from .abstract_no_approx_posterior import NoApproxPosterior
import torch




class ContrastiveDivergence(NoApproxPosterior):
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


        dic_total.update(self.train_approx_posterior(x, step))
        dic_total.update(self.train_approx_posterior_reverse(x, z_g_k, step))

        dic_total.update({
            "loss_e": loss_e,
            "loss_g": loss_g,
            "mse_loss": mse_loss,
            "en_pos": en_pos,
            "en_neg": en_neg,
            })
         
        
        return dic_total