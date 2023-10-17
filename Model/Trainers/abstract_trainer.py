import math
import torch
import torch.nn as nn
import tqdm
import time
import wandb
import os
import numpy as np

from omegaconf import OmegaConf

from ..Generator import AbstractGenerator
from ..Encoder import AbstractEncoder
from ..Energy import get_energy_network
from ..Optim import get_optimizer, grad_clipping
from ..Optim.Schedulers import get_scheduler
from ..Prior import get_prior, get_extra_prior
from ..Sampler import get_posterior_sampler, get_prior_sampler
from ..Utils.log_utils import log, draw, get_extremum, plot_contour
from ..Utils.aggregate_posterior import AggregatePosterior

class AbstractTrainer:
    def __init__(
        self,
        cfg,
    ) -> None:

        self.cfg = cfg

        self.generator = AbstractGenerator(cfg)
        self.energy = get_energy_network(cfg.energy.network_name, cfg.trainer.nz, cfg.energy.ndf)
        self.prior = get_prior(cfg.trainer.nz, cfg.prior).to(cfg.trainer.device)
        self.extra_prior = get_extra_prior(cfg.trainer.nz, cfg.extra_prior).to(cfg.trainer.device)
        self.encoder = AbstractEncoder(cfg, cfg.trainer.nz, cfg.dataset.nc)

        self.sampler_prior = get_prior_sampler(cfg.sampler_prior)
        self.sampler_posterior = get_posterior_sampler(cfg.sampler_posterior)
        self.mse = nn.MSELoss(reduction="sum")
        self.mse_test = nn.MSELoss(reduction='none')
        self.proposal = torch.distributions.normal.Normal(
            torch.tensor(cfg.trainer.proposal_mean, device=cfg.trainer.device, dtype=torch.float32),
            torch.tensor(cfg.trainer.proposal_std, device=cfg.trainer.device, dtype=torch.float32),)
        self.log_var_p = torch.tensor(0, device=cfg.trainer.device, dtype=torch.float32)
        if cfg.trainer.log_dir is None:
            cfg.trainer.log_dir = os.path.join(cfg.machine.root, "logs",)
            print("Setting log dir to " + cfg.trainer.log_dir)
        self.logger = wandb.init(
            project="LatentEBM_{}".format(cfg.dataset.dataset_name),
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.trainer.log_dir,
            name= str(cfg.trainer.nz)+ "_" + cfg.prior.prior_name + "_" + cfg.trainer.trainer_name + time.strftime("%Y%m%d-%H%M%S"),
        )
        self.n_iter = cfg.trainer.n_iter
        self.n_iter_pretrain = cfg.trainer.n_iter_pretrain
        self.compile()

    def compile(self):
        self.generator.to(self.cfg.trainer.device)
        self.opt_generator = get_optimizer(self.cfg.optim_generator, self.generator)
        self.sch_generator = get_scheduler(self.cfg.scheduler_generator, self.opt_generator)

        self.encoder.to(self.cfg.trainer.device)
        self.opt_encoder = get_optimizer( self.cfg.optim_encoder, self.encoder)
        self.sch_encoder = get_scheduler(self.cfg.scheduler_encoder, self.opt_encoder)
        
        self.energy.to(self.cfg.trainer.device)
        self.opt_energy = get_optimizer(self.cfg.optim_energy, self.energy)
        self.sch_energy = get_scheduler(self.cfg.scheduler_energy, self.opt_energy)

        self.extra_prior.to(self.cfg.trainer.device)
        self.opt_prior = get_optimizer(self.cfg.optim_prior, self.extra_prior)
        self.sch_prior = get_scheduler(self.cfg.scheduler_prior, self.opt_prior)



    def train(self, train_dataloader, val_dataloader=None, test_dataloader=None):
        self.global_step = 0
        iterator = iter(train_dataloader)
        for self.global_step in tqdm.tqdm(range(self.n_iter_pretrain + self.n_iter)):
            try :
                x = next(iterator)[0].to(self.cfg.trainer.device)
            except StopIteration:
                self.sch_encoder.step()
                self.sch_energy.step()
                self.sch_generator.step()
                self.sch_prior.step()
                iterator = iter(train_dataloader)
                x = next(iterator)[0].to(self.cfg.trainer.device)
            x = x.reshape(x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)



            if self.global_step < self.n_iter_pretrain:
                dic_loss = self.train_step_standard_elbo(x, self.global_step)
            else:
                dic_loss = self.train_step(x, self.global_step)

            # Log
            if self.global_step % self.cfg.trainer.log_every == 0:
                # for key, item in dic_loss.items():
                    # dic_loss[key] = item.
                log(self.global_step, dic_loss, logger=self.logger)
            # Save
            if self.global_step % self.cfg.trainer.save_images_every == 0:
                self.draw_samples(x, self.global_step)
                self.plot_latent(dataloader=train_dataloader,step = self.global_step)

            # Eval
            if self.global_step % self.cfg.trainer.val_every == 0 and val_dataloader is not None:
                self.eval(val_dataloader, self.global_step)
                # self.SNIS_eval(val_dataloader, self.global_step)
            # Test
            if self.global_step%self.cfg.trainer.test_every == 0 and test_dataloader is not None :
                self.eval(test_dataloader, self.global_step, name="test/")
                # self.SNIS_eval(test_dataloader, self.global_step, name="test/")

            


    def train_step_standard_elbo(self, x, step):
        self.opt_generator.zero_grad()
        self.opt_energy.zero_grad()
        self.opt_encoder.zero_grad()
        dic_total = {}

        z_e_0, z_g_0 = self.prior.sample(x.shape[0]), self.prior.sample(x.shape[0])
        param = self.encoder(x)
        dic_param, dic_param_feedback = self.encoder.latent_distribution.get_params(param)
        dic_total.update(dic_param_feedback)

        # Reparametrization trick
        z_q = self.encoder.latent_distribution.r_sample(param, dic_params=dic_param).reshape(x.shape[0], self.cfg.trainer.nz)


        # Reconstruction loss
        x_hat = self.generator(z_q)
        mse_loss = self.mse(x_hat, x) / x.shape[0]

        loss_g = self.generator.get_loss(x_hat, x).reshape(x.shape[0]).mean(dim=0)

        # KL without ebm
        KL_loss = self.encoder.latent_distribution.calculate_kl(self.prior, param, z_q, dic_params=dic_param).mean(dim=0)

        # Entropy posterior
        entropy_posterior = self.encoder.latent_distribution.calculate_entropy(param, dic_params=dic_param).mean(dim=0)


        # ELBO
        loss_total = loss_g + KL_loss
        loss_total.backward()

        dic_total.update({
            "loss_g": loss_g.item(),
            "KL_loss": KL_loss.item(),
            "elbo": -loss_total.item(),
            "mse_loss": mse_loss.item(),
            "entropy_posterior": entropy_posterior.item(),
        })
        dic_total.update(dic_param)
        self.opt_energy.step()
        self.opt_generator.step()
        self.opt_encoder.step()

        return dic_total


    def test(self, data):
        raise NotImplementedError

    def train_step(self, x, step):
        raise NotImplementedError
    
    def grad_clipping_all_net(self, liste_name = [], step=None):
        for net_name in liste_name:
            if not hasattr(self.cfg, "optim_"+net_name) or hasattr(self, "opt"+net_name):
                raise ValueError("cfg.optim_{} does not exist".format(net_name))
            else :
                current_optim_cfg = getattr(self.cfg,"optim_"+net_name)
                net = getattr(self, net_name)
                current_optim = getattr(self, "opt_"+net_name)

        grad_clipping(net, net_name, current_optim_cfg, current_optim, self.logger, step=step)

    

    def log_partition_estimate(self, step, name="val/"):
        with torch.no_grad():
            batch_size_val = self.cfg.dataset.batch_size_val
            sampled = 0
            log_partition_estimate = 0
            total_energy = 0
            for k in range(int(np.ceil(self.cfg.trainer.nb_sample_partition_estimate_val/batch_size_val))):
                z_e_0 = self.prior.sample(self.cfg.trainer.nb_sample_partition_estimate_val,)[:self.cfg.trainer.nb_sample_partition_estimate_val-sampled]
                sampled+=z_e_0.shape[0]
                current_energy = self.energy(z_e_0).flatten(1).sum(1)
                total_energy += self.energy(z_e_0).sum(0)
                log_partition_estimate += torch.logsumexp(-current_energy,0) 
            log_partition_estimate = log_partition_estimate - math.log(sampled)
            log(step, {"log_z":log_partition_estimate.item()}, logger=self.logger, name=name)
            log(step, {"energy_base_dist":(total_energy/sampled).item()}, logger=self.logger, name=name)
        return log_partition_estimate



    def eval(self, val_data, step, name="val/"):
        with torch.no_grad():
            log_partition_estimate = self.log_partition_estimate(step, name=name)
            iterator = iter(val_data)
            for i in range(len(val_data)):
                dic_feedback = {}
                batch = next(iterator)
                x = batch[0].to(self.cfg.trainer.device)
                x = x.reshape(x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)

                x_expanded = x.unsqueeze(0).expand(self.cfg.trainer.multiple_sample_val, x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size).flatten(0,1)
                expanded_batch_size = x.shape[0]*self.cfg.trainer.multiple_sample_val
                dic = {}
                param = self.encoder(x)
                dic_param, dic_param_feedback = self.encoder.latent_distribution.get_params(param)
                dic_feedback.update(dic_param_feedback)


                # Reparam trick
                z_q = self.encoder.latent_distribution.r_sample(param, n_samples = self.cfg.trainer.multiple_sample_val, dic_params=dic_param).reshape(expanded_batch_size, self.cfg.trainer.nz)
                x_hat = self.generator(z_q)

                # Reconstruction loss :
                loss_g = self.generator.get_loss(x_hat, x_expanded).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0]).mean(dim=0)
                mse_loss = self.mse_test(x_hat, x_expanded).reshape(self.cfg.trainer.multiple_sample_val, x.shape[0], -1).sum(dim=2).mean(dim=0)


                # KL without ebm
                z_q_no_multiple = z_q.reshape(self.cfg.trainer.multiple_sample_val, x.shape[0], self.cfg.trainer.nz)[0]
                KL_loss = self.encoder.latent_distribution.calculate_kl(self.prior, param, z_q_no_multiple, dic_params=dic_param)

                # Entropy posterior
                entropy_posterior = self.encoder.latent_distribution.calculate_entropy(param, dic_params=dic_param)

                # Gaussian extra_prior
                log_prob_extra_prior = self.extra_prior.log_prob(z_q)
                log_prob_extra_prior = log_prob_extra_prior.reshape(self.cfg.trainer.multiple_sample_val,x.shape[0])
               
                # Energy :
                energy_approximate = self.energy(z_q).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0])
                base_dist_z_approximate = self.prior.log_prob(z_q).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0])

                
                # Different loss :
                loss_ebm = (energy_approximate + log_partition_estimate.exp() - 1).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0]).mean(dim=0)
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

                multi_gaussian = self.extra_prior.log_prob(z_q).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])

                # Different Weights :
                posterior_distribution = self.encoder.latent_distribution.log_prob(param, z_q, dic_params=dic_param).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                log_weights_energy = (energy_approximate + base_dist_z_approximate - posterior_distribution).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                log_weights_no_energy = (base_dist_z_approximate - posterior_distribution).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                log_weights_gaussian = (multi_gaussian - posterior_distribution).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                log_weights_energy = torch.log_softmax(log_weights_energy, dim=0)
                log_weights_no_energy = torch.log_softmax(log_weights_no_energy, dim=0)
                log_weights_gaussian = torch.log_softmax(log_weights_gaussian, dim=0)



                # Reconstruction loss :
                loss_g = self.generator.get_loss(x_hat, x_expanded).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                SNIS_energy = (log_weights_energy+loss_g).logsumexp(0).reshape(x.shape[0])
                SNIS_no_energy = (log_weights_no_energy+loss_g).logsumexp(0).reshape(x.shape[0])
                SNIS_extra_prior = (log_weights_gaussian+loss_g).logsumexp(0).reshape(x.shape[0])
                
                dic_feedback.update({
                    "SNIS_energy": -SNIS_energy,
                    "SNIS_no_energy" : -SNIS_no_energy,
                    "SNIS_extra_prior" : -SNIS_extra_prior
                })
               

                for key, value in dic_feedback.items():
                    if key not in dic:
                        dic[key] = []
                    dic[key].append(value.reshape(x.shape[0]))

            
            for key in dic:
                dic[key] = torch.stack(dic[key], dim=0).mean().item()
            log(step, dic, logger=self.logger, name=name)



    def draw_samples(self, x, step):
        batch_save = min(64, x.shape[0])
        z_e_0, z_g_0 = self.prior.sample(batch_save), self.prior.sample(batch_save)
        z_e_k, z_grad_norm = self.sampler_prior(z_e_0, self.energy, self.prior,)
        z_g_k, z_g_grad_norm, z_e_grad_norm = self.sampler_posterior(z_g_0,x[:batch_save], self.generator, self.energy, self.prior,)

        with torch.no_grad():
            norm_z_e_0 = z_e_0.norm(dim=1).reshape(batch_save,)
            norm_z_e_k = z_e_k.norm(dim=1).reshape(batch_save,)
            norm_z_g_k = z_g_0.norm(dim=1).reshape(batch_save,)
            
            # Mean norm
            log(step, {"norm_z_e_0":norm_z_e_0.mean().item()}, self.logger, name="sample/")
            log(step, {"norm_z_e_k":norm_z_e_k.mean().item()}, self.logger, name="sample/")
            log(step, {"norm_z_g_k":norm_z_g_k.mean().item()}, self.logger, name="sample/")
            # Std Norm
            log(step, {"norm_z_e_0_std":norm_z_e_0.std().item()}, self.logger, name="sample/")
            log(step, {"norm_z_e_k_std":norm_z_e_k.std().item()}, self.logger, name="sample/")
            log(step, {"norm_z_g_k_std":norm_z_g_k.std().item()}, self.logger, name="sample/")


            x_base, mu_base =self.generator.sample(z_e_0, return_mean=True)
            draw(x_base.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleBaseDistribution")
            draw(mu_base.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanBaseDistribution")
        
            x_prior, mu_prior =self.generator.sample(z_e_k, return_mean=True)
            draw(x_prior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleEBMPrior")
            draw(mu_prior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanEBMPrior")
    
            x_posterior, mu_posterior =self.generator.sample(z_g_k, return_mean=True)
            draw(x_posterior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleEBMPosterior")
            draw(mu_posterior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanEBMPosterior")


            param = self.encoder(x[:batch_save])
            if self.cfg.encoder.latent_distribution_name == "gaussian":
                sample_mean=param.chunk(2, 1)[0].reshape(-1, self.cfg.trainer.nz, 1, 1)
            elif self.cfg.encoder.latent_distribution_name == "uniform":
                dic_param, _ = self.encoder.latent_distribution.get_params(param)
                min_aux, max_aux = dic_param['min'], dic_param['max']
                sample_mean = (min_aux + max_aux)/2

            x_reconstruction, mu_reconstruction =self.generator.sample(sample_mean, return_mean=True)
            draw(x_reconstruction.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleReconstruction")
            draw(mu_reconstruction.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanReconstruction")
        
            extra_prior_samples = self.extra_prior.sample(batch_save).reshape(batch_save, self.cfg.trainer.nz, 1, 1).to(self.cfg.trainer.device)
            x_prior, mu_prior = self.generator.sample(extra_prior_samples, return_mean=True)
            draw(x_prior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleExtraPrior")
            draw(mu_prior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanExtraPrior")


    def plot_latent(self, dataloader, step):
        if self.cfg.trainer.nz != 2:
            pass
        else :
            data = next(iter(dataloader))[0].to(self.cfg.trainer.device)
            while len(data)<1000:
                data = torch.cat([data, next(iter(dataloader))[0].to(self.cfg.trainer.device)], dim=0)
            data = data[:1000]

            len_samples = min(1000, data.shape[0])
            params = self.encoder(data[:len_samples])
            

            if self.cfg.encoder.latent_distribution_name == "gaussian":
                mu_q = params.chunk(2,1)[0].reshape(len_samples, 2)
            elif self.cfg.encoder.latent_distribution_name == "uniform":
                dic_param, _ = self.encoder.latent_distribution.get_params(params)
                min_aux, max_aux = dic_param['min'], dic_param['max']
                mu_q = (min_aux + max_aux)/2

                
            z_e_0, z_g_0 = self.prior.sample(len_samples), self.prior.sample(len_samples)
            z_e_k, z_grad_norm = self.sampler_prior(z_e_0, self.energy, self.prior,)
            z_g_k, z_g_grad_norm, z_e_grad_norm = self.sampler_posterior(z_g_0, data[:len_samples], self.generator, self.energy, self.prior,)


            if self.cfg.prior.prior_name == "gaussian":
                energy_list_small_scale, energy_list_names, x, y = self.get_all_energies(z_e_0, min_x=-3, max_x=3, params=params)
                samples_aux = self.cut_samples(z_e_0, min_x=-3, max_x=3)
                plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Base Distribution SC")
                
                samples_aux = self.cut_samples(z_e_k, min_x=-3, max_x=3)
                plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Prior SC")


                samples_aux = self.cut_samples(mu_q, min_x=-3, max_x=3)
                plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Approximate Posterior SC")

                samples_aux = self.cut_samples(z_g_k, min_x=-3, max_x=3)
                plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Posterior SC")


                energy_list_large_scale, energy_list_names, x, y = self.get_all_energies(z_e_0, min_x=-10, max_x=10, params=params)
                samples_aux = self.cut_samples(z_e_0, min_x=-10, max_x=10)
                plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Base Distribution LC")
                
                samples_aux = self.cut_samples(z_e_k, min_x=-10, max_x=10)
                plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Prior LC")

                samples_aux = self.cut_samples(mu_q, min_x=-10, max_x=10)
                plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Approximate Posterior LC")

                samples_aux = self.cut_samples(z_g_k, min_x=-10, max_x=10)
                plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Posterior LC")


                energy_list_large_scale, energy_list_names, x, y = self.get_all_energies(z_e_0, min_x=-30, max_x=30, params=params)
                samples_aux = self.cut_samples(z_e_0, min_x=-30, max_x=30)
                plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Base Distribution XLC")
                
                samples_aux = self.cut_samples(z_e_k, min_x=-30, max_x=30)
                plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Prior XLC")

                samples_aux = self.cut_samples(mu_q, min_x=-30, max_x=30)
                plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Approximate Posterior XLC")

                samples_aux = self.cut_samples(z_g_k, min_x=-30, max_x=30)
                plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Posterior XLC")


            elif self.cfg.prior.prior_name =='uniform' :
                energy_list_small_scale, energy_list_names, x, y = self.get_all_energies(z_e_0, min_x=self.cfg.prior.min, max_x=self.cfg.prior.max, params=params)
                samples_aux = self.cut_samples(z_e_0, min_x=-3, max_x=3)
                plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Base Distribution SC")
                
                samples_aux = self.cut_samples(z_e_k, min_x=-3, max_x=3)
                plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Prior SC")


                samples_aux = self.cut_samples(mu_q, min_x=-3, max_x=3)
                plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Approximate Posterior SC")

                samples_aux = self.cut_samples(z_g_k, min_x=-3, max_x=3)
                plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Posterior SC")
            else :
                raise ValueError("Prior name not recognized")



    def cut_samples(self, samples, min_x=-10, max_x =-10):
        min_y = min_x
        max_y = max_x
        tensor_min = torch.cat([torch.full_like(samples[:,0,None], min_x),torch.full_like(samples[:,1, None], min_y)], dim=1)
        tensor_max = torch.cat([torch.full_like(samples[:,0,None], max_x),torch.full_like(samples[:,1, None], max_y)], dim=1)
        samples = torch.where(samples < tensor_min, tensor_min, samples)
        samples = torch.where(samples > tensor_max, tensor_max, samples)
        return samples


    def get_all_energies(self, samples, min_x=-10, max_x=-10, params=None):
        samples_mean = samples.mean(0)
        samples_std = samples.std(0)
        min_y = min_x
        max_y = max_x
        
        grid_coarseness = self.cfg.trainer.grid_coarseness
        device = self.cfg.trainer.device


        x = np.linspace(min_x, max_x, grid_coarseness)
        y = np.linspace(min_y, max_y, grid_coarseness)
        xx, yy = np.meshgrid(x, y)
        xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
        xy = torch.from_numpy(xy).float().to(device)

        energy_base_dist = - self.prior.log_prob(xy).reshape(grid_coarseness,grid_coarseness,)
        energy_extra_prior = - self.extra_prior.log_prob(xy).reshape(grid_coarseness,grid_coarseness)
        just_energy = self.energy(xy).reshape(grid_coarseness, grid_coarseness)
       
        energy_prior = just_energy + energy_base_dist

        energy_list = [energy_base_dist, energy_prior, energy_extra_prior, just_energy]
        energy_list_names = ["Base Distribution", "EBM Prior", "Extra Prior", "Just EBM"]

        if params is not None and self.cfg.prior.prior_name == "gaussian": # Does not work with uniform distribution
            dic_params, _ = self.encoder.latent_distribution.get_params(params)
            dist_posterior = self.encoder.latent_distribution.get_distribution(params, dic_params=dic_params)
            aggregate = AggregatePosterior(dist_posterior, params.shape[0])
            aggregate_energy = -aggregate.log_prob(xy).reshape(grid_coarseness, grid_coarseness)
            energy_list.append(aggregate_energy)
            energy_list_names.append("Aggregate Posterior")

        return energy_list, energy_list_names, x, y