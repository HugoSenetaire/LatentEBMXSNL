import math

import torch


from ..Regularization import regularization_encoder, regularization

from .abstract_trainer import AbstractTrainer
from ..Utils import AggregatePosterior






class SNELBO_Aggregate(AbstractTrainer):
    def __init__(self, cfg, ):
        super().__init__(cfg, )
        self.detach_approximate_posterior = cfg.trainer.detach_approximate_posterior

    def train_step(self, x, step):
        dic_feedback = {}
        self.opt_generator.zero_grad()
        self.opt_energy.zero_grad()
        self.opt_encoder.zero_grad()

        z_e_0, z_g_0 = self.prior.sample(x.shape[0]), self.prior.sample(x.shape[0])
        param = self.encoder(x)
        dic_param, dic_param_feedback = self.encoder.latent_distribution.get_params(param)
        dic_feedback.update(dic_param_feedback)
        aggregate = AggregatePosterior(self.encoder.latent_distribution.get_distribution(params=param, dic_params=dic_param),  x.shape[0])


        # Reparam trick
        z_q = self.encoder.latent_distribution.r_sample(param).reshape(x.shape[0], self.cfg.trainer.nz)
        z_agg = aggregate.sample(x.shape[0])
        x_hat = self.generator(z_q)

        # Reconstruction loss :
        loss_g = self.generator.get_loss(x_hat, x).reshape(x.shape[0]).mean(dim=0)

        # KL without ebm
        KL_loss = self.encoder.latent_distribution.calculate_kl(self.prior, param, z_q, dic_params=dic_param, empirical_kl=self.cfg.trainer.empirical_kl).mean(dim=0)

        # Entropy posterior
        entropy_posterior = self.encoder.latent_distribution.calculate_entropy(param, dic_params=dic_param, empirical_entropy=self.cfg.trainer.empirical_entropy).mean(dim=0)

 
        # Energy :
        if self.cfg.trainer.detach_approximate_posterior:
            z_q = z_q.detach()
        energy_approximate = self.energy(z_q).reshape(x.shape[0])
        energy_base_dist = self.energy(z_e_0).reshape(x.shape[0])
        energy_aggregate = self.energy(z_agg).reshape(x.shape[0])

        base_dist_z_approximate = self.prior.log_prob(z_q).reshape(x.shape[0])
        base_dist_z_base_dist = self.prior.log_prob(z_e_0).reshape(x.shape[0])
        base_dist_z_aggregate = self.prior.log_prob(z_agg).reshape(x.shape[0])

        aggregate_z_approximate = aggregate.log_prob(z_q).reshape(x.shape[0])
        aggregate_z_base_dist = aggregate.log_prob(z_e_0).reshape(x.shape[0])
        aggregate_z_aggregate = aggregate.log_prob(z_agg).reshape(x.shape[0])


        log_partition_estimate = torch.logsumexp(-energy_aggregate,0) - math.log(energy_aggregate.shape[0])
        loss_ebm = (energy_approximate).mean() + log_partition_estimate.exp() -1

        loss_total = loss_g + KL_loss + loss_ebm


        dic_regul = regularization(self.energy, z_q, z_e_0, energy_approximate, energy_aggregate, self.cfg, self.logger, step)
        for key, item in dic_regul.items():
            loss_total += item
        dic_feedback.update(dic_regul)
        dic_regul_encoder = regularization_encoder(dic_param, self.encoder, self.cfg, self.logger, step)
        for key, item in dic_regul_encoder.items():
            loss_total += item
        dic_feedback.update(dic_regul_encoder)

        
        
        
        
        loss_total.backward()
        self.grad_clipping_all_net(["energy", "generator", "encoder"], step)

        dic_feedback.update({
            "loss_g":loss_g.item(),
            "entropy_posterior":entropy_posterior.item(),
            "loss_ebm": loss_ebm.item(),
            "base_dist_z_approximate": base_dist_z_approximate.mean().item(),
            "base_dist_z_base_dist" : base_dist_z_base_dist.mean().item(),
            "base_dist_z_aggregate" : base_dist_z_aggregate.mean().item(),
            "log_Z":log_partition_estimate.item(),
            "KL_loss_no_ebm": KL_loss.item(),
            "energy_approximate": energy_approximate.mean().item(),
            "energy_base_dist": energy_base_dist.mean().item(),
            "energy_aggregate": energy_aggregate.mean().item(),
            "aggregate_z_approximate": aggregate_z_approximate.mean().item(),
            "aggregate_z_base_dist" : aggregate_z_base_dist.mean().item(),
            "aggregate_z_aggregate" : aggregate_z_aggregate.mean().item(),
            "approx_elbo" : -loss_total.item(),
            "elbo_no_ebm" : -loss_g.item() - KL_loss.item(),
        })

        self.opt_energy.step()
        if not self.cfg.trainer.fix_generator :
            self.opt_generator.step()
        if not self.cfg.trainer.fix_encoder :
            self.opt_encoder.step()
        return dic_feedback
    