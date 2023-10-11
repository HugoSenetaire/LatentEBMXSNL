import math

from ..Regularization.regularizer_ebm import regularization
from .abstract_trainer import AbstractTrainer




class ContrastiveDivergence(AbstractTrainer):
    def __init__(self, cfg, ):
        super().__init__(cfg, )
        
    def train_step(self, x, step):
        z_e_0, z_g_0 = self.base_dist.sample(self.cfg.dataset.batch_size), self.base_dist.sample(self.cfg.dataset.batch_size)
        z_e_k = self.sampler_prior(z_e_0, self.energy, self.base_dist,)
        z_g_k = self.sampler_posterior(z_g_0, x, self.generator, self.energy, self.base_dist,)


        self.opt_generator.zero_grad()
        x_hat = self.generator(z_g_k.detach())
        loss_g = self.generator.get_loss(x_hat, x).mean(dim=0).mean()
        mse_loss = self.mse(x_hat, x) / x.shape[0]
        loss_g.backward()
        self.opt_generator.step()
        self.opt_energy.zero_grad()
        en_pos, en_neg = self.energy(z_g_k.detach()).mean(), self.energy(z_e_k.detach()).mean()
        loss_e = en_pos - en_neg
        dic_loss = regularization(self.energy, z_g_k, z_e_k, en_pos, en_neg, self.cfg, self.logger, step)
        for key, item in dic_loss.items():
            loss_e += item
        loss_e.backward()
        self.opt_energy.step()
        dic_loss = {
            "loss_e": loss_e,
            "loss_g": loss_g,
            "mse_loss": mse_loss,
            "en_pos": en_pos,
            "en_neg": en_neg,
            }
        
        return dic_loss