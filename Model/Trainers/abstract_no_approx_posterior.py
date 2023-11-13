import math
import torch
import tqdm

from ..Regularization import regularization_encoder

from .abstract_trainer import AbstractTrainer
from ..Encoder import AbstractEncoder
from ..Optim import get_optimizer, get_scheduler
from ..Utils.log_utils import log


class NoApproxPosterior(AbstractTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.reverse_encoder = AbstractEncoder(cfg, cfg.trainer.nz, cfg.dataset.nc, reverse=True)
        self.compile_reverse()
        self.use_reverse = self.cfg.trainer.use_reverse

    def save_model(self, name=""):
        super().save_model(name)
        torch.save(self.reverse_encoder.state_dict(), self.cfg.machine.root + "/reverse_encoder.pth")

    def load_model(self, name=""):
        super().load_model(name)
        self.reverse_encoder.load_state_dict(torch.load(self.cfg.machine.root + "/reverse_encoder.pth"))
        self.compile_reverse()
    
    def compile_reverse(self):
        self.reverse_encoder.to(self.cfg.trainer.device)
        assert self.encoder != self.reverse_encoder
        self.opt_reverse_encoder = get_optimizer(self.cfg.optim_encoder, self.reverse_encoder)
        assert self.opt_reverse_encoder != self.opt_encoder
        self.sch_reverse_encoder = get_scheduler(self.cfg.scheduler_encoder, self.opt_reverse_encoder)
        assert self.sch_reverse_encoder != self.sch_encoder

    def train_approx_posterior_reverse(self, x, z_g_k, step):
        if not self.use_reverse:
            return {}
        dic_feedback = {}
        self.opt_reverse_encoder.zero_grad()
        params_reverse = self.reverse_encoder(x)
        dic_params_reverse, dic_params_feedback_reverse = self.reverse_encoder.latent_distribution.get_params(params_reverse)
        dic_feedback.update(dic_params_feedback_reverse)

        # KL 
        kl_posterior_reverse = -self.reverse_encoder.latent_distribution.log_prob(params_reverse, z_g_k.detach(), dic_params=dic_params_reverse).mean(dim=0)
        loss_kl_reverse = kl_posterior_reverse.mean()
        loss_kl_reverse.backward()

        # Entropy posterior
        entropy_posterior_reverse = self.reverse_encoder.latent_distribution.calculate_entropy(params_reverse, dic_params=dic_params_reverse, empirical_entropy=self.cfg.trainer.empirical_entropy).mean(dim=0)


        # Reparam trick
        z_q = self.reverse_encoder.latent_distribution.r_sample(params_reverse, dic_params=dic_params_reverse).reshape(x.shape[0], self.cfg.trainer.nz)
        x_hat = self.generator(z_q)

        # Reconstruction loss :
        loss_g = self.generator.get_loss(x_hat, x).reshape(x.shape[0]).mean(dim=0)

        # KL without ebm
        KL_loss = self.reverse_encoder.latent_distribution.calculate_kl(self.prior, params_reverse, z_q, dic_params=dic_params_reverse, empirical_kl=self.cfg.trainer.empirical_kl).mean(dim=0)

        # Entropy posterior
        entropy_posterior = self.reverse_encoder.latent_distribution.calculate_entropy(params_reverse, dic_params=dic_params_reverse, empirical_entropy=self.cfg.trainer.empirical_entropy).mean(dim=0)


        self.opt_reverse_encoder.step()
        dic_feedback.update({
            "kl_loss": kl_posterior_reverse.mean().item(),
            "entropy_posterior": entropy_posterior_reverse.mean().item(),
            "loss_g": loss_g.mean().item(),
            "KL_loss_no_ebm": KL_loss.mean().item(),
            "entropy_posterior": entropy_posterior.mean().item(),
            "mse_mu_z_g_k": (params_reverse.chunk(2,1)[0]-z_g_k).pow(2).sum(dim=1).mean(dim=0).item(),
        })

        return {key+"_reverse":value for key, value in dic_feedback.items()}


    def train_approx_posterior(self, x, step):
        if not self.forward_posterior:
            return {}
        dic_feedback = {}
        # Train the encoder to go to regular ebm, not really the same thing, it's just so that I get better approximation, could do reverse KL ?
        self.opt_encoder.zero_grad()
        params = self.encoder(x)


        dic_params, dic_params_feedback = self.encoder.latent_distribution.get_params(params)
        dic_feedback.update(dic_params_feedback)

        # Reparam trick
        z_q = self.encoder.latent_distribution.r_sample(params, dic_params=dic_params).reshape(x.shape[0], self.cfg.trainer.nz)
        x_hat = self.generator(z_q)


        # Reconstruction loss :
        loss_g = self.generator.get_loss(x_hat, x).reshape(x.shape[0]).mean(dim=0)

        # KL without ebm
        KL_loss = self.encoder.latent_distribution.calculate_kl(self.prior, params, z_q, dic_params=dic_params, empirical_kl=self.cfg.trainer.empirical_kl).mean(dim=0)

        # Entropy posterior
        entropy_posterior = self.encoder.latent_distribution.calculate_entropy(params, dic_params=dic_params, empirical_entropy=self.cfg.trainer.empirical_entropy).mean(dim=0)


        loss_elbo = loss_g + KL_loss
        dic_regul_encoder = regularization_encoder(dic_params, self.encoder, self.cfg, self.logger, step=step)
        for key, item in dic_regul_encoder.items():
            loss_elbo += item
        dic_feedback.update(dic_regul_encoder)



        loss_elbo.backward()
        self.grad_clipping_all_net(["encoder"], step=step)

        self.opt_encoder.step()
        dic_feedback.update({
           "elbo_loss": -loss_elbo.mean().item(),
            "kl_loss": KL_loss.mean().item(),
            "entropy_posterior": entropy_posterior.mean().item(),
        })

        return dic_feedback
        


    def SNIS_eval(self, val_data, step, name="val/"):
        super().SNIS_eval(val_data, step, name=name)
        if self.use_reverse :
            iterator = iter(val_data)
            total_dic_feedback = {}
            ranger = tqdm.tqdm(range(len(val_data)), desc=f"snis_{name[:-1]}_reverse", position=1, leave=False)
            multiple_sample_val_SNIS = getattr(self.cfg.trainer, "multiple_sample_{}_SNIS".format(name[:-1]),)
            for i in ranger:
                dic_feedback = {}
                batch = next(iterator)
                x = batch[0].to(self.cfg.trainer.device)
                x = x.reshape(x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)

                x_expanded = x.unsqueeze(0).expand(multiple_sample_val_SNIS, x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size).flatten(0,1)
                expanded_batch_size = x.shape[0]*multiple_sample_val_SNIS

                param = self.reverse_encoder(x)
                dic_param, dic_param_feedback = self.reverse_encoder.latent_distribution.get_params(param)
                dic_param_feedback = {key+"_reverse":value for key, value in dic_param_feedback.items()}
                dic_feedback.update(dic_param_feedback)

                # Reparam trick
                z_q = self.reverse_encoder.latent_distribution.r_sample(param, n_samples = multiple_sample_val_SNIS, dic_params=dic_param).reshape(expanded_batch_size, self.cfg.trainer.nz)
                x_hat = self.generator(z_q)

                multi_gaussian = self.extra_prior.log_prob(z_q).reshape(multiple_sample_val_SNIS,x.shape[0])

                # Energy :
                energy_approximate = self.energy(z_q).reshape(multiple_sample_val_SNIS,x.shape[0])
                base_dist_z_approximate = self.prior.log_prob(z_q).reshape(multiple_sample_val_SNIS,x.shape[0])

                # Different Weights :
                posterior_distribution = self.reverse_encoder.latent_distribution.log_prob(param, z_q, dic_params=dic_param).reshape(multiple_sample_val_SNIS,x.shape[0])
                log_weights_energy = (energy_approximate + base_dist_z_approximate - posterior_distribution).reshape(multiple_sample_val_SNIS,x.shape[0])
                log_weights_no_energy = (base_dist_z_approximate - posterior_distribution).reshape(multiple_sample_val_SNIS,x.shape[0])
                log_weights_gaussian = (multi_gaussian - posterior_distribution).reshape(multiple_sample_val_SNIS,x.shape[0])
                log_weights_energy = torch.log_softmax(log_weights_energy, dim=0)
                log_weights_no_energy = torch.log_softmax(log_weights_no_energy, dim=0)
                log_weights_gaussian = torch.log_softmax(log_weights_gaussian, dim=0)



                # Reconstruction loss :
                loss_g = self.generator.get_loss(x_hat, x_expanded).reshape(multiple_sample_val_SNIS,x.shape[0])
                SNIS_energy = (log_weights_energy+loss_g).logsumexp(0).reshape(x.shape[0])
                SNIS_no_energy = (log_weights_no_energy+loss_g).logsumexp(0).reshape(x.shape[0])
                SNIS_extra_prior = (log_weights_gaussian+loss_g).logsumexp(0).reshape(x.shape[0])
                
                dic_feedback.update({
                    "SNIS_energy": -SNIS_energy,
                    "SNIS_no_energy_reverse" : -SNIS_no_energy,
                    "SNIS_extra_prior_reverse" : -SNIS_extra_prior
                })

                for key, value in dic_feedback.items():
                    if key+"_reverse" not in total_dic_feedback:
                        total_dic_feedback[key+"_reverse"] = []
                    total_dic_feedback[key+"_reverse"].append(value.reshape(x.shape[0]))
            
            for key in total_dic_feedback:
                total_dic_feedback[key] = torch.cat(total_dic_feedback[key], dim=0).mean().item()
            log(step, total_dic_feedback, logger=self.logger, name=name)

    def elbo_eval(self, val_data, log_partition_estimate, step, name="val/"):
        super().elbo_eval(val_data,log_partition_estimate, step, name=name)
        if self.use_reverse:
            iterator = iter(val_data)
            total_dic_feedback = {}
            ranger = tqdm.tqdm(range(len(val_data)), desc=f"elbo_{name[:-1]}_reverse", position=1, leave=False)
            multiple_sample_val_elbo = getattr(self.cfg.trainer, "multiple_sample_{}".format(name[:-1]),)
            for i in ranger:
                dic_feedback = {}
                batch = next(iterator)
                x = batch[0].to(self.cfg.trainer.device)
                x = x.reshape(x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)

                x_expanded = x.unsqueeze(0).expand(multiple_sample_val_elbo, x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size).flatten(0,1)
                expanded_batch_size = x.shape[0]*multiple_sample_val_elbo

                param = self.reverse_encoder(x)
                dic_param, dic_param_feedback = self.reverse_encoder.latent_distribution.get_params(param)
                dic_feedback.update(dic_param_feedback)

                # Reparam trick
                z_q = self.reverse_encoder.latent_distribution.r_sample(param, n_samples = multiple_sample_val_elbo, dic_params=dic_param).reshape(expanded_batch_size, self.cfg.trainer.nz)
                x_hat = self.generator(z_q)

                # Reconstruction loss :
                loss_g = self.generator.get_loss(x_hat, x_expanded).reshape(multiple_sample_val_elbo,x.shape[0]).mean(dim=0)
                mse_loss = self.mse_test(x_hat, x_expanded).reshape(multiple_sample_val_elbo, x.shape[0], -1).sum(dim=2).mean(dim=0)


                # KL without ebm
                z_q_no_multiple = z_q.reshape(multiple_sample_val_elbo, x.shape[0], self.cfg.trainer.nz)[0]
                KL_loss = self.reverse_encoder.latent_distribution.calculate_kl(self.prior, param, z_q_no_multiple, dic_params=dic_param)

                # Entropy posterior
                entropy_posterior = self.reverse_encoder.latent_distribution.calculate_entropy(param, dic_params=dic_param)


                # Gaussian extra_prior
                log_prob_extra_prior = self.extra_prior.log_prob(z_q)
                log_prob_extra_prior = log_prob_extra_prior.reshape(multiple_sample_val_elbo,x.shape[0])
                
                # Energy :
                energy_approximate = self.energy(z_q).reshape(multiple_sample_val_elbo,x.shape[0])
                base_dist_z_approximate = self.prior.log_prob(z_q).reshape(multiple_sample_val_elbo,x.shape[0])

                
                # Different loss :
                loss_ebm = (energy_approximate + log_partition_estimate.exp() - 1).reshape(multiple_sample_val_elbo,x.shape[0]).mean(dim=0)
                loss_total = loss_g + KL_loss + loss_ebm
                elbo_extra_prior = -loss_g + entropy_posterior + (log_prob_extra_prior).mean(dim=0)


                
                dic_feedback.update({
                    "loss_g": loss_g,
                    "entropy_posterior": entropy_posterior,
                    "loss_ebm": loss_ebm,
                    "base_dist_z_approximate": base_dist_z_approximate.mean(dim=0),
                    "KL_loss_no_ebm": KL_loss,
                    "energy_approximate": energy_approximate.mean(dim=0),
                    "approx_elbo": -loss_total,
                    "elbo_no_ebm": -loss_g - KL_loss,
                    "elbo_extra_prior": elbo_extra_prior,
                    "mse_loss": mse_loss,
                    "log_prob_extra_prior_z_approximate": log_prob_extra_prior.mean(dim=0),
                })


                for key, value in dic_feedback.items():
                    if key+"_reverse" not in total_dic_feedback:
                        total_dic_feedback[key+"_reverse"] = []
                    total_dic_feedback[key+"_reverse"].append(value.reshape(x.shape[0]))
            
            for key in total_dic_feedback:
                total_dic_feedback[key] = torch.cat(total_dic_feedback[key], dim=0).mean().item()
            log(step, total_dic_feedback, logger=self.logger, name=name)





