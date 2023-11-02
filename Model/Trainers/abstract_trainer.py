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
from ..Utils.utils_fid import calculate_frechet

class AbstractTrainer:
    def __init__(
        self,
        cfg,
    ) -> None:

        self.cfg = cfg

        self.generator = AbstractGenerator(cfg).to(cfg.trainer.device)
        self.energy = get_energy_network(cfg.energy.network_name, cfg.trainer.nz, cfg.energy.ndf)
        self.prior = get_prior(cfg.trainer.nz, cfg.prior).to(cfg.trainer.device)
        self.extra_prior = get_extra_prior(cfg.trainer.nz, cfg.extra_prior).to(cfg.trainer.device)
        self.encoder = AbstractEncoder(cfg, cfg.trainer.nz, cfg.dataset.nc).to(cfg.trainer.device)

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
            project="LatentEBM_{}_{}".format(cfg.dataset.dataset_name,str(cfg.trainer.nz)),
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.trainer.log_dir,
            name= cfg.trainer.trainer_name + "_" + cfg.prior.prior_name + "_" + cfg.encoder.latent_distribution_name + time.strftime("%Y%m%d-%H%M%S"),
        )
        self.n_iter = cfg.trainer.n_iter
        self.n_iter_pretrain = cfg.trainer.n_iter_pretrain
        self.n_iter_pretrain_encoder = cfg.trainer.n_iter_pretrain_encoder
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
        for self.global_step in tqdm.tqdm(range(self.n_iter_pretrain_encoder + self.n_iter_pretrain + self.n_iter)):
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


            if self.global_step < self.n_iter_pretrain_encoder:
                dic_loss = self.train_step_only_encoder(x, self.global_step)
            elif self.global_step < self.n_iter_pretrain + self.n_iter_pretrain_encoder:
                dic_loss = self.train_step_standard_elbo(x, self.global_step)
            else:
                dic_loss = self.train_step(x, self.global_step)

            # Log
            if self.global_step % self.cfg.trainer.log_every == 0:
                log(self.global_step, dic_loss, logger=self.logger)

            # Save
            if self.global_step % self.cfg.trainer.save_images_every == 0:
                self.draw_samples(x, self.global_step)
                self.plot_latent(dataloader=train_dataloader,step = self.global_step)


            # Eval
            if (self.global_step) % self.cfg.trainer.val_every == 0 and val_dataloader is not None:
                self.eval(val_dataloader, self.global_step)
                self.fid_eval(val_data=val_dataloader, step=self.global_step, name="val/")

                
            # Test
            if (self.global_step)%self.cfg.trainer.test_every == 0 and test_dataloader is not None and self.global_step>1:
                self.eval(test_dataloader, self.global_step, name="test/")

            

    def train_step_only_encoder(self, x, step):
        self.opt_generator.zero_grad()
        self.opt_energy.zero_grad()
        self.opt_encoder.zero_grad()
        dic_total = {}
        param = self.encoder(x)
        dic_param, dic_param_feedback = self.encoder.latent_distribution.get_params(param)
        dic_total.update(dic_param_feedback)

        # Reparametrization trick
        z_q = self.encoder.latent_distribution.r_sample(param, dic_params=dic_param).reshape(x.shape[0], self.cfg.trainer.nz)

        # KL without ebm
        KL_loss = self.encoder.latent_distribution.calculate_kl(self.prior, param, z_q, dic_params=dic_param).mean(dim=0)

        # Entropy posterior
        entropy_posterior = self.encoder.latent_distribution.calculate_entropy(param, dic_params=dic_param).mean(dim=0)


        # ELBO
        loss_total = KL_loss
        loss_total.backward()

        dic_total.update({
            "KL_loss": KL_loss.item(),
            "entropy_posterior": entropy_posterior.item(),
        })
        dic_total.update(dic_param)
        self.opt_energy.step()
        self.opt_generator.step()
        self.opt_encoder.step()

        return dic_total

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
            if net_name == "reverse_encoder":
                current_optim_cfg = getattr(self.cfg,"optim_encoder")
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
            nb_sample_partition = getattr(self.cfg.trainer, "nb_sample_partition_estimate_{}".format(name[:-1]))
            for k in range(int(np.ceil(nb_sample_partition/batch_size_val))):
                z_e_0 = self.prior.sample(nb_sample_partition,)[:nb_sample_partition-sampled]
                sampled+=z_e_0.shape[0]
                current_energy = self.energy(z_e_0).flatten(1).sum(1)
                total_energy += self.energy(z_e_0).sum(0)
                log_partition_estimate += torch.logsumexp(-current_energy,0) 
            log_partition_estimate = log_partition_estimate - math.log(sampled)
            log(step, {"log_z":log_partition_estimate.item()}, logger=self.logger, name=name)
            log(step, {"energy_base_dist":(total_energy/sampled).item()}, logger=self.logger, name=name)
        return log_partition_estimate


    def get_partition_estimate(self, step, name="val/"):
        log_partition_estimate = self.log_partition_estimate(step, name=name)
        return log_partition_estimate

    def eval(self, val_data, step, name="val/"):
        with torch.no_grad():
            dic_feedback = {}
            self.generator.eval()
            self.encoder.eval()
            self.energy.eval()
            self.extra_prior.eval()
            log_partition_estimate = self.get_partition_estimate(step, name=name)
            self.elbo_eval(val_data, log_partition_estimate, step, name=name)
            self.SNIS_eval(val_data=val_data, step=step, name=name)


    def elbo_eval(self, val_data, log_partition_estimate, step, name="val/"):
        iterator = iter(val_data)
        total_dic_feedback = {}
        ranger = tqdm.tqdm(range(len(val_data)), desc=f"elbo_{name[:-1]}", position=1, leave=False)
        multiple_sample_val_elbo = getattr(self.cfg.trainer, "multiple_sample_{}".format(name[:-1]))
        for i in ranger:
            dic_feedback = {}
            batch = next(iterator)
            x = batch[0].to(self.cfg.trainer.device)
            x = x.reshape(x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)

            x_expanded = x.unsqueeze(0).expand(multiple_sample_val_elbo, x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size).flatten(0,1)
            expanded_batch_size = x.shape[0]*multiple_sample_val_elbo

            param = self.encoder(x)
            dic_param, dic_param_feedback = self.encoder.latent_distribution.get_params(param)
            dic_feedback.update(dic_param_feedback)

            # Reparam trick
            z_q = self.encoder.latent_distribution.r_sample(param, n_samples = multiple_sample_val_elbo, dic_params=dic_param).reshape(expanded_batch_size, self.cfg.trainer.nz)
            if torch.any(torch.isnan(z_q)):
                print("z_q nan")
            x_hat = self.generator(z_q)


            if hasattr(self.encoder.latent_distribution, "reverse"):
                param_reverse = self.encoder.latent_distribution.reverse(param)
                dic_param_reverse, dic_param_feedback_reverse = self.encoder.latent_distribution.get_params(param_reverse)
                z_q_reverse = self.encoder.latent_distribution.r_sample(param_reverse, dic_params=dic_param_reverse).reshape(expanded_batch_size, self.cfg.trainer.nz)
                x_hat_reverse = self.generator(z_q_reverse)

            # Reconstruction loss :
            loss_g = self.generator.get_loss(x_hat, x_expanded).reshape(multiple_sample_val_elbo,x.shape[0]).mean(dim=0)
            mse_loss = self.mse_test(x_hat, x_expanded).reshape(multiple_sample_val_elbo, x.shape[0], -1).sum(dim=2).mean(dim=0)


            # KL without ebm
            z_q_no_multiple = z_q.reshape(multiple_sample_val_elbo, x.shape[0], self.cfg.trainer.nz)[0]
            KL_loss = self.encoder.latent_distribution.calculate_kl(self.prior, param, z_q_no_multiple, dic_params=dic_param)

            # Entropy posterior
            entropy_posterior = self.encoder.latent_distribution.calculate_entropy(param, dic_params=dic_param)


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
                if key not in total_dic_feedback:
                    total_dic_feedback[key] = []
                total_dic_feedback[key].append(value.reshape(x.shape[0]))
        
        for key in total_dic_feedback:
            total_dic_feedback[key] = torch.cat(total_dic_feedback[key], dim=0).mean().item()
        log(step, total_dic_feedback, logger=self.logger, name=name)



    def SNIS_eval(self, val_data, step, name="val/"):

        iterator = iter(val_data)
        total_dic_feedback = {}
        ranger = tqdm.tqdm(range(len(val_data)), desc=f"SNIS_{name[:-1]}", position=1, leave=False)
        multiple_sample_val_SNIS = getattr(self.cfg.trainer, "multiple_sample_{}_SNIS".format(name[:-1]))
        for i in ranger:
            dic_feedback = {}
            batch = next(iterator)
            x = batch[0].to(self.cfg.trainer.device)
            x = x.reshape(x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)

            x_expanded = x.unsqueeze(0).expand(multiple_sample_val_SNIS, x.shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size).flatten(0,1)
            expanded_batch_size = x.shape[0]*multiple_sample_val_SNIS

            param = self.encoder(x)
            dic_param, dic_param_feedback = self.encoder.latent_distribution.get_params(param)
            dic_feedback.update(dic_param_feedback)

            # Reparam trick
            z_q = self.encoder.latent_distribution.r_sample(param, n_samples = multiple_sample_val_SNIS, dic_params=dic_param).reshape(expanded_batch_size, self.cfg.trainer.nz)
            x_hat = self.generator(z_q)

            multi_gaussian = self.extra_prior.log_prob(z_q).reshape(multiple_sample_val_SNIS,x.shape[0])

            # Energy :
            energy_approximate = self.energy(z_q).reshape(multiple_sample_val_SNIS,x.shape[0])
            base_dist_z_approximate = self.prior.log_prob(z_q).reshape(multiple_sample_val_SNIS,x.shape[0])

            # Different Weights :
            posterior_distribution = self.encoder.latent_distribution.log_prob(param, z_q, dic_params=dic_param).reshape(multiple_sample_val_SNIS,x.shape[0])
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
                "SNIS_no_energy" : -SNIS_no_energy,
                "SNIS_extra_prior" : -SNIS_extra_prior
            })

            for key, value in dic_feedback.items():
                if key not in total_dic_feedback:
                    total_dic_feedback[key] = []
                total_dic_feedback[key].append(value.reshape(x.shape[0]))
        
        for key in total_dic_feedback:
            total_dic_feedback[key] = torch.cat(total_dic_feedback[key], dim=0).mean().item()
        log(step, total_dic_feedback, logger=self.logger, name=name)
        

    def fid_eval(self, val_data, step, name="val/"):
        batch_save = min(256, val_data.dataset.__len__())
        z_e_0 = self.prior.sample(batch_save)
        z_e_k, _ = self.sampler_prior(z_e_0, self.energy, self.prior,)
        x_sample, x_mean = self.generator.sample(z_e_k, return_mean=True)
        x_sample = x_sample.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size).to('cpu')
        x_mean = x_mean.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size).to('cpu')
        val_data = torch.stack([val_data.dataset.__getitem__(i)[0] for i in np.random.randint(0, val_data.dataset.__len__(), batch_save)]).to('cpu')

        fid = calculate_frechet(val_data, x_sample,)
        fid_mean = calculate_frechet(val_data, x_mean,)
        log(step, {"fid":fid}, logger=self.logger, name=name)
        log(step, {"fid_mean":fid_mean}, logger=self.logger, name=name)


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
    
            x_posterior, mu_posterior = self.generator.sample(z_g_k, return_mean=True)
            draw(x_posterior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleEBMPosterior")
            draw(mu_posterior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanEBMPosterior")


            param = self.encoder(x[:batch_save])
            sample_mean, sample_std = self.encoder.latent_distribution.get_plots(param,)

            x_reconstruction, mu_reconstruction = self.generator.sample(sample_mean, return_mean=True)
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
            mu_q, log_var_q = self.encoder.latent_distribution.get_plots(params)
            samples_approx_post = self.encoder.latent_distribution.r_sample(params, n_samples=10).reshape(10*len_samples, self.cfg.trainer.nz)

            z_e_0, z_g_0 = self.prior.sample(len_samples), self.prior.sample(len_samples)
            z_e_k, z_grad_norm = self.sampler_prior(z_e_0, self.energy, self.prior,)
            z_g_k, z_g_grad_norm, z_e_grad_norm = self.sampler_posterior(z_g_0, data[:len_samples], self.generator, self.energy, self.prior,)
         
            liste_samples = [z_e_0, z_e_k, mu_q, samples_approx_post, z_g_k]
            liste_samples_name = ["Latent Base Distribution", "Latent Prior", "Latent Approximate Posterior Mu ", "Latent Approximate Posterior Sample", "Latent Posterior"]


            if hasattr(self, "reverse_encoder"):
                params_reverse = self.reverse_encoder(data[:len_samples])
                mu_q_reverse, log_var_q_reverse = self.reverse_encoder.latent_distribution.get_plots(params_reverse)
                liste_samples.append(mu_q_reverse)
                liste_samples_name.append("Latent Approximate Posterior Reverse Mu")
                samples_approx_reverse_post = self.reverse_encoder.latent_distribution.r_sample(params_reverse, n_samples=10).reshape(10*len_samples, self.cfg.trainer.nz)
                liste_samples.append(samples_approx_reverse_post)
                liste_samples_name.append("Latent Approximate Posterior Reverse Sample")
            else :
                params_reverse = None

            if self.cfg.prior.prior_name == "gaussian":
                self.plot_samples_2d(liste_samples, -3, 3, liste_samples_name, step, params=params, params_reverse=params_reverse)
                self.plot_samples_2d(liste_samples, -10, 10, liste_samples_name, step, params=params, params_reverse=params_reverse)
                self.plot_samples_2d(liste_samples, -30, 30, liste_samples_name, step, params=params, params_reverse=params_reverse)
            elif "hyperspherical" in self.cfg.prior.prior_name:
                self.plot_samples_2d(liste_samples, -2, 2, liste_samples_name, step, params=params, params_reverse=params_reverse)
            elif self.cfg.prior.prior_name =='uniform' :
                self.plot_samples_2d(liste_samples, self.cfg.prior.min+1e-2, self.cfg.prior.max-1e-2, liste_samples_name, step, params=params, params_reverse=params_reverse)
            else :
                raise ValueError("Prior name not recognized")

    def plot_samples_2d(self, samples, min_x, max_x, liste_samples_name, step, params = None, params_reverse=None, ):
        energy_list_small_scale, energy_list_names, x, y = self.get_all_energies(samples[0], min_x=min_x, max_x=max_x, params = params, params_reverse = params_reverse)
        for sample, samples_names in zip(samples, liste_samples_name):
            samples_aux = self.cut_samples(sample, min_x=min_x, max_x=max_x)
            title = samples_names+f" [{str(min_x)}, {str(max_x)}]"
            plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, title=title, logger=self.logger,)


    def cut_samples(self, samples, min_x=-10, max_x =-10):
        min_y = min_x
        max_y = max_x
        tensor_min = torch.cat([torch.full_like(samples[:,0,None], min_x),torch.full_like(samples[:,1, None], min_y)], dim=1)
        tensor_max = torch.cat([torch.full_like(samples[:,0,None], max_x),torch.full_like(samples[:,1, None], max_y)], dim=1)
        
        samples = torch.where(samples < tensor_min, tensor_min, samples)
        samples = torch.where(samples > tensor_max, tensor_max, samples)
        return samples


    def get_all_energies(self, samples, min_x=-10, max_x=-10, params=None, params_reverse=None):
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
        # if params is not None and self.cfg.encoder.latent_distribution_name == 'gaussian' :
        if params is not None and self.cfg.encoder.latent_distribution_name != 'uniform' and "mises" not in self.cfg.encoder.latent_distribution_name:
        # == "gaussian": # Does not work with uniform distribution
            dic_params, _ = self.encoder.latent_distribution.get_params(params)
            dist_posterior = self.encoder.latent_distribution.get_distribution(params, dic_params=dic_params)
            aggregate = AggregatePosterior(dist_posterior, params.shape[0], device = device)
            aggregate_energy = -aggregate.log_prob(xy).reshape(grid_coarseness, grid_coarseness)
            energy_list.append(aggregate_energy)
            energy_list_names.append("Aggregate Posterior")
            
        if params_reverse is not None and self.cfg.encoder.latent_distribution_name != 'uniform' and "mises" not in self.cfg.encoder.latent_distribution_name:
        # if params_reverse is not None and self.cfg.encoder.latent_distribution_name == 'gaussian':
            dic_params, _ = self.reverse_encoder.latent_distribution.get_params(params)
            dist_posterior = self.reverse_encoder.latent_distribution.get_distribution(params, dic_params=dic_params)
            aggregate = AggregatePosterior(dist_posterior, params.shape[0], device = device)
            aggregate_energy = -aggregate.log_prob(xy).reshape(grid_coarseness, grid_coarseness)
            energy_list.append(aggregate_energy)
            energy_list_names.append("Aggregate Posterior Reverse")

        return energy_list, energy_list_names, x, y