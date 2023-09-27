import math

import torch

from grad_clipper import grad_clipping_all_net
from regularization import regularization
from sampler import (sample_langevin_posterior, sample_langevin_prior,
                     sample_p_0, sample_p_data)

from .abstract_trainer import AbstractTrainer


class TrainerCDTrick(AbstractTrainer):
    def __init__(self, cfg,):
        super().__init__(cfg,)

    def train_step(self, x, step):

        z_e_0, z_g_0 = self.base_dist.sample(), self.base_dist.sample()
        z_e_k = sample_langevin_prior(z_e_0, self.E, self.cfg["K_0"], self.cfg["a_0"])
        z_g_k = sample_langevin_posterior(z_g_0, x, self.G, self.E, self.cfg["K_1"], self.cfg["a_1"], self.loss_reconstruction)


        self.optG.zero_grad()
        x_hat = self.G(z_g_k.detach())
        loss_g = self.loss_reconstruction(x_hat, x).mean()
        loss_g.backward()
        grad_clipping_all_net([self.G], ["G"], [self.optG], self.logger, self.cfg, step=step)
        self.optG.step()



        self.optE.zero_grad()
        # en_pos, en_neg = E(z_g_k.detach()).mean(), E(z_e_k.detach()).mean()
        energy_posterior = self.E(z_g_k.detach()).flatten(1).sum(1)
        z_proposal = self.proposal.sample(z_g_k.shape)
        energy_proposal = self.E(z_proposal.detach()).flatten(1).sum(1)
        base_dist_z_proposal = self.base_dist.log_prob(z_proposal.flatten(1)).sum(1)
        base_dist_z_posterior = self.base_dist.log_prob(z_g_k.flatten(1)).sum(1)
        base_dist_z_base_dist = self.base_dist.log_prob(z_e_0.flatten(1)).sum(1)
        proposal_z_proposal = self.proposal.log_prob(z_proposal.flatten(1)).sum(1)
        proposal_z_posterior = self.proposal.log_prob(z_g_k.flatten(1)).sum(1)
        proposal_z_base_dist = self.proposal.log_prob(z_e_0.flatten(1)).sum(1)


        log_partition_estimate = torch.logsumexp(-energy_proposal,0) - math.log(energy_proposal.shape[0])
        loss_e = (energy_posterior-proposal_z_posterior).mean() + log_partition_estimate.exp() -1
        regularization(self.E, z_g_k, z_proposal, energy_posterior, energy_proposal, self.cfg, self.logger, step=step)
        loss_e.backward()
        grad_clipping_all_net([self.E], ["E"], [self.optE], self.logger, self.cfg, step=step)
        self.optE.step()
        dic_loss = {
            "loss_e": loss_e.mean().item(),
            "loss_g": loss_g.mean().item(),
            "base_dist_z_proposal": base_dist_z_proposal.mean().item(),
            "base_dist_z_posterior":base_dist_z_posterior.mean().item(),
            "base_dist_z_base_dist": base_dist_z_base_dist.mean().item(),
            "proposal_z_proposal": proposal_z_proposal.mean().item(),
            "proposal_z_posterior":proposal_z_posterior.mean().item(),
            "proposal_z_base_dist": proposal_z_base_dist.mean().item(),
            "en_pos": energy_posterior.mean().item(),
            "en_neg": energy_proposal.mean().item(),
            "log_z" : log_partition_estimate.item()
        }

        return dic_loss