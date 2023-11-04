import math

import torch

from ..Regularization import regularization, regularization_encoder

from ..Utils import log
from .abstract_no_approx_posterior import NoApproxPosterior


class ContrastiveDivergenceLogTrick(NoApproxPosterior):
    def __init__(self, cfg,):
        super().__init__(cfg,)



    def train_step(self, x, step):
        dic_feedback = {}

        z_e_0, z_g_0 = self.prior.sample(x.shape[0]), self.prior.sample(x.shape[0])
        z_e_k = self.sampler_prior(z_e_0, self.energy, self.prior,)
        z_g_k = self.sampler_posterior(z_g_0, x, self.generator, self.energy, self.prior,)

        # z_e_k, z_grad_norm = self.sampler_prior(z_e_0, self.energy, self.prior,)
        # z_g_k, z_g_grad_norm, z_e_grad_norm = self.sampler_posterior(z_g_0, x, self.generator, self.energy, self.prior,)

        z_e_0_norm = z_e_0.detach().norm(dim=1).mean()
        z_e_k_norm = z_e_k.detach().norm(dim=1).mean()
        z_g_k_norm = z_g_k.detach().norm(dim=1).mean()

        dic_feedback.update({
            "z_e_0_norm": z_e_0_norm,
            "z_e_k_norm": z_e_k_norm,
            "z_g_k_norm": z_g_k_norm,
            "lr_e": self.opt_energy.param_groups[0]["lr"],
            "lr_g": self.opt_generator.param_groups[0]["lr"],
        })


        self.opt_generator.zero_grad()
        x_hat = self.generator(z_g_k.detach())
        loss_g = self.generator.get_loss(x_hat, x).mean(dim=0).mean()
        loss_g.backward()
        self.grad_clipping_all_net(["generator"], step=step)
        self.opt_generator.step()



        self.opt_energy.zero_grad()
        # en_pos, en_neg = E(z_g_k.detach()).mean(), E(z_e_k.detach()).mean()
        energy_posterior = self.energy(z_g_k.detach())
        z_proposal = self.proposal.sample(z_g_k.shape)
        energy_proposal = self.energy(z_proposal.detach())

        base_dist_z_proposal = self.prior.log_prob(z_proposal)
        base_dist_z_posterior = self.prior.log_prob(z_g_k)
        base_dist_z_base_dist = self.prior.log_prob(z_e_0)
        
        proposal_z_proposal = self.proposal.log_prob(z_proposal)
        proposal_z_posterior = self.proposal.log_prob(z_g_k)
        proposal_z_base_dist = self.proposal.log_prob(z_e_0)


        log_partition_estimate = torch.logsumexp(-energy_proposal,0) - math.log(energy_proposal.shape[0])
        loss_e = (energy_posterior-proposal_z_posterior).mean() + log_partition_estimate.exp() -1
        dic_regul = regularization(self.energy, z_g_k, z_proposal, energy_posterior, energy_proposal, self.cfg, self.logger, step=step)
        for key, item in dic_regul.items():
            loss_e += item
        dic_feedback.update(dic_regul)
        loss_e.backward()
        self.grad_clipping_all_net(["energy"], step=step)

        self.opt_energy.step()

        dic_feedback.update(self.train_approx_posterior(x, step))
        if self.use_reverse :
            dic_feedback.update(self.train_approx_posterior_reverse(x, z_g_k, step))





        dic_feedback.update({
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
            "log_z" : log_partition_estimate.item(),
         
        })

        return dic_feedback
    
