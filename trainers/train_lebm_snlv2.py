import math

import torch

from grad_clipper import grad_clipping_all_net
from regularization import regularization
from sampler import (sample_langevin_posterior, sample_langevin_prior,
                     sample_p_0, sample_p_data)

from .abstract_trainer import AbstractTrainer


class Trainer_LEBM_SNL2(AbstractTrainer):
    def __init__(self, cfg, ):
        super().__init__(cfg, )

    def train_step(self, x, step):
        """
        Here in the case where the base distribution is actually the proposal, then
        i can just calculate the entropy of the posterior, then I get loss ebm on the same level, hopefully
        """

        fix_encoder = self.cfg["fix_encoder"]
        fix_decoder= self.cfg["fix_decoder"]

        self.optG.zero_grad()
        self.optE.zero_grad()
        self.optEncoder.zero_grad()

        z_e_0 = self.base_dist.sample((self.cfg["batch_size"],self.cfg['nz'],1,1))
        z_g_0 = self.base_dist.sample((self.cfg["batch_size"],self.cfg["nz"],1,1))
        mu_q, log_var_q = self.Encoder(x).chunk(2,1)
        std_q = torch.exp(0.5*log_var_q)

        # Reparam trick
        eps = torch.randn_like(mu_q)
        z_q = (eps.mul(std_q).add_(mu_q)).reshape(-1,self.cfg["nz"],1,1)
        x_hat = self.G(z_q)


        # Reconstruction loss :
        loss_g = self.loss_reconstruction(x_hat, x).mean(dim=0)

        # KL without ebm
        KL_loss = 0.5 * (self.log_var_p - log_var_q -1 +  (log_var_q.exp() + mu_q.pow(2))/self.log_var_p.exp())
        KL_loss = KL_loss.sum(dim=1).mean(dim=0)

        # Entropy posterior
        entropy_posterior = torch.sum(0.5* (math.log(2*math.pi) +  log_var_q + 1), dim=1).mean()

        # Energy :
        z_proposal = self.proposal.sample((z_e_0.shape[0], self.cfg["nz"],1,1))
        energy_approximate = self.E(z_q).flatten(1).sum(1)
        energy_proposal = self.E(z_proposal).flatten(1).sum(1)
        energy_prior = self.E(z_e_0).flatten(1).sum(1)

        base_dist_z_proposal = self.base_dist.log_prob(z_proposal.flatten(1)).sum(1)
        base_dist_z_posterior = self.base_dist.log_prob(z_q.flatten(1)).sum(1)
        base_dist_z_base_dist = self.base_dist.log_prob(z_e_0.flatten(1)).sum(1)
        proposal_z_proposal = self.proposal.log_prob(z_proposal.flatten(1)).sum(1)
        proposal_z_posterior = self.proposal.log_prob(z_q.flatten(1)).sum(1)
        proposal_z_base_dist = self.proposal.log_prob(z_e_0.flatten(1)).sum(1)


        log_partition_estimate = torch.logsumexp(-energy_proposal - proposal_z_proposal,0) - math.log(energy_proposal.shape[0])
        loss_ebm = energy_approximate.mean() + log_partition_estimate.exp() -1
        # loss_ebm =  energy_approximate.mean() + log_partition_estimate

        loss_total = loss_g + KL_loss + loss_ebm
        regularization(self.E, z_q, z_e_0, energy_approximate, energy_proposal, self.cfg, self.logger, step)
        loss_total.backward()
        grad_clipping_all_net([self.E,self.G,self.Encoder], ["E", "G", "Encoder"], [self.optE, self.optG, self.optEncoder,], self.logger, self.cfg, step)

        dic_loss = {
            "loss_g": loss_g.item(),
            "entropy_posterior": entropy_posterior.item(),
            "loss_ebm": loss_ebm.item(),
            "log_Z": log_partition_estimate.item(),
            "KL_loss_no_ebm": KL_loss.item(),
            "energy_approximate": energy_approximate.mean().item(),
            "energy_proposal" : energy_proposal.mean().item(),
            "energy_prior": energy_prior.mean().item(),
            "base_dist_z_proposal": base_dist_z_proposal.mean().item(),
            "base_dist_z_posterior":base_dist_z_posterior.mean().item(),
            "base_dist_z_base_dist": base_dist_z_base_dist.mean().item(),
            "proposal_z_proposal": proposal_z_proposal.mean().item(),
            "proposal_z_posterior":proposal_z_posterior.mean().item(),
            "proposal_z_base_dist": proposal_z_base_dist.mean().item(),
            "approx_elbo" : -loss_total.item(),
        }

        self.optE.step()
        if not fix_decoder :
            self.optG.step()
        if not fix_encoder :
            self.optEncoder.step()


        return dic_loss