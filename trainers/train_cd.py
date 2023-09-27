import math
from sampler import sample_langevin_posterior, sample_langevin_prior
from grad_clipper import grad_clipping_all_net
from regularization import regularization
from .abstract_trainer import AbstractTrainer


class TrainerCD(AbstractTrainer):
    def __init__(self, cfg, ):
        super().__init__(cfg, )

    def train_step(self, x, step):
        z_e_0, z_g_0 = self.base_dist.sample(), self.base_dist.sample()
        z_e_k = sample_langevin_prior(z_e_0, self.E, self.cfg["K_0"], self.cfg["a_0"])
        z_g_k = sample_langevin_posterior(z_g_0, x, self.G, self.E, self.cfg["K_1"], self.cfg["a_1"], self.cfg["llhd_sigma"], self.loss_reconstruction)


        self.optG.zero_grad()
        x_hat = self.G(z_g_k.detach())
        loss_g = self.loss_reconstruction(x_hat, x).mean()
        loss_g.backward()
        grad_clipping_all_net([self.G], ["G"], [self.optG], self.logger, self.cfg, step=step)
        self.optG.step()
        self.optE.zero_grad()
        en_pos, en_neg = self.E(z_g_k.detach()).mean(), self.E(z_e_k.detach()).mean()
        loss_e = en_pos - en_neg
        regularization(self.E, z_g_k, z_e_k, en_pos, en_neg, self.cfg, self.logger, step)
        loss_e.backward()
        grad_clipping_all_net([self.E], ["E"], [self.optE], self.logger, self.cfg, step)
        self.optE.step()
        dic_loss = {
            "loss_e": loss_e,
            "loss_g": loss_g,
            "en_pos": en_pos,
            "en_neg": en_neg,
            }
        
        return dic_loss