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
from ..Prior import get_prior, GaussianPrior
from ..Sampler.sampler import SampleLangevinPosterior, SampleLangevinPrior, Sampler
from ..Utils.log_utils import log, draw, get_extremum, plot_contour

class AbstractTrainer:
    def __init__(
        self,
        cfg,
    ) -> None:

        self.cfg = cfg

        self.generator = AbstractGenerator(cfg)
        self.energy = get_energy_network(cfg.energy.network_name, cfg.trainer.nz, cfg.energy.ndf)
        self.prior = get_prior(cfg)
        self.encoder = AbstractEncoder(cfg, cfg.trainer.nz, cfg.dataset.nc)

        self.sampler_prior = SampleLangevinPrior(self.cfg.sampler_prior.K, self.cfg.sampler_prior.a,)
        self.sampler_posterior = SampleLangevinPosterior(self.cfg.sampler_posterior.K, self.cfg.sampler_posterior.a, )
        
        self.proposal = torch.distributions.normal.Normal(
            torch.tensor(cfg.trainer.proposal_mean, device=cfg.trainer.device, dtype=torch.float32),
            torch.tensor(cfg.trainer.proposal_std, device=cfg.trainer.device, dtype=torch.float32),)
        self.base_dist = GaussianPrior(cfg)
        self.log_var_p = torch.tensor(0, device=cfg.trainer.device, dtype=torch.float32)
        if cfg.trainer.log_dir is None:
            cfg.trainer.log_dir = os.path.join(cfg.machine.root, "logs",)
            print("Setting log dir to " + cfg.trainer.log_dir)
        self.logger = wandb.init(
            project="LatentEBM",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.trainer.log_dir,
            name= str(cfg.trainer.nz)+ "_" + cfg.dataset.dataset_name + "_" + cfg.trainer.trainer_name + time.strftime("%Y%m%d-%H%M%S"),
        )
        self.n_iter = cfg.trainer.n_iter
        self.n_iter_pretrain = cfg.trainer.n_iter_pretrain
        self.compile()

    def compile(self):

        self.generator.to(self.cfg.trainer.device)
        self.opt_generator = get_optimizer(self.cfg.optim_generator, self.generator)

        self.encoder.to(self.cfg.trainer.device)
        self.opt_encoder = get_optimizer( self.cfg.optim_encoder, self.encoder)
        
        self.energy.to(self.cfg.trainer.device)
        self.opt_energy = get_optimizer(self.cfg.optim_energy, self.energy)

        self.prior.to(self.cfg.trainer.device)
        self.opt_prior = get_optimizer(self.cfg.optim_prior, self.prior)



    def train(self, train_data, val_data=None):
        self.sampler = Sampler(train_data)
        for i in tqdm.tqdm(range(self.n_iter_pretrain + self.n_iter)):
            x = self.sampler.sample_p_data(self.cfg.dataset.batch_size).to(self.cfg.trainer.device)
            if i < self.n_iter_pretrain:
                dic_loss = self.train_step_standard_elbo(x, i)
            else:
                dic_loss = self.train_step(x, i)

            # Log
            if i % self.cfg.trainer.log_every == 0:
                log(i, dic_loss, logger=self.logger)
            # Save
            if i % self.cfg.trainer.save_images_every == 0:
                self.draw_samples(x, i)
                self.plot_latent(i)

            # Eval
            if i % self.cfg.trainer.val_every == 0 and val_data is not None:
                self.eval(val_data, i)
                self.SNIS_eval(val_data, i)
            # Test
            if i%self.cfg.trainer.test_every == 0 and val_data is not None :
                self.eval(val_data, i, name="test/")
                self.SNIS_eval(val_data, i, name="test/")


    def train_step_standard_elbo(self, x, step):
        self.opt_generator.zero_grad()
        self.opt_energy.zero_grad()
        self.opt_encoder.zero_grad()

        z_e_0, z_g_0 = self.base_dist.sample(self.cfg.dataset.batch_size), self.base_dist.sample(self.cfg.dataset.batch_size)
        mu_q, log_var_q = self.encoder(x).chunk(2, 1)

        # Reparametrization trick
        std_q = torch.exp(0.5 * log_var_q)
        eps = torch.randn_like(mu_q)
        z_q = (eps.mul(std_q).add_(mu_q)).reshape(x.shape[0], self.cfg.trainer.nz,)

        # Reconstruction loss
        x_hat = self.generator(z_q)
        mse = nn.MSELoss(reduction="sum")
        loss_g = self.generator.get_loss(x_hat, x).reshape(x.shape[0]).mean(dim=0)

        # KL loss
        KL_loss = 0.5 * (
            self.log_var_p
            - log_var_q
            - 1
            + (log_var_q.exp() + mu_q.pow(2)) / self.log_var_p.exp()
        )
        KL_loss = KL_loss.sum(dim=1).mean(dim=0)

        # ELBO
        loss_total = loss_g + KL_loss
        loss_total.backward()

        dic_loss = {
            "loss_g": loss_g.item(),
            "KL_loss": KL_loss.item(),
            "elbo": -loss_total.item(),
        }
        self.opt_energy.step()
        self.opt_generator.step()
        self.opt_encoder.step()

        return dic_loss


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

    


    def SNIS_eval(self, val_data, step, name="val/"):
        with torch.no_grad():
            k=0
            dic = {}
            if name == "val/":
                max_len = len(val_data)/10
            else :
                max_len = len(val_data)
            while k*self.cfg.dataset.batch_size_val<max_len:
                x = val_data[k*self.cfg.dataset.batch_size_val:(k+1)*self.cfg.dataset.batch_size_val].to(self.cfg.trainer.device)
                x_expanded = x.unsqueeze(0).expand(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0],self.cfg.dataset.nc,self.cfg.dataset.img_size,self.cfg.dataset.img_size).flatten(0,1)
                expanded_batch_size = x.shape[0]*self.cfg.trainer.multiple_sample_val_SNIS
                
                mu_q, log_var_q = self.encoder(x_expanded).chunk(2,1)
                std_q = torch.exp(0.5*log_var_q)
                epsilon = torch.randn_like(mu_q)
                z_q = (epsilon.mul(std_q).add_(mu_q)).reshape(expanded_batch_size, self.cfg.trainer.nz,)
                x_hat = self.generator(z_q)

                energy_prior = self.energy(z_q).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                base_dist = self.base_dist.log_prob(z_q).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                multi_gaussian = self.prior.log_prob(z_q).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])


                posterior_distribution = torch.distributions.normal.Normal(mu_q, std_q).log_prob(z_q.flatten(1)).sum(1).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                weights_energy = (energy_prior + base_dist - posterior_distribution).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                weights_no_energy = (base_dist - posterior_distribution).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                weights_gaussian = (multi_gaussian - posterior_distribution).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                weights_energy = torch.softmax(weights_energy, dim=0)
                weights_no_energy = torch.softmax(weights_no_energy, dim=0)
                weights_gaussian = torch.softmax(weights_gaussian, dim=0)



                # Reconstruction loss :
                loss_g = self.generator.get_loss(x_hat, x_expanded).reshape(self.cfg.trainer.multiple_sample_val_SNIS,x.shape[0])
                SNIS_energy = (weights_energy*loss_g).sum(0)
                SNIS_no_energy = (weights_no_energy*loss_g).sum(0)
                SNIS_gaussian = (weights_gaussian*loss_g).sum(0)
                
                dic_loss = {
                    "SNIS_energy": -SNIS_energy,
                    "SNIS_no_energy" : -SNIS_no_energy,
                    "SNIS_gaussian" : -SNIS_gaussian
                }
                k = k + 1

                for key, value in dic_loss.items():
                    if key not in dic:
                        dic[key] = []
                    dic[key].append(value.reshape(x.shape[0]))
            for key in dic:
                dic[key] = torch.cat(dic[key],dim=0).mean().item()
            log(step, dic, logger=self.logger, name=name)




    def eval(self, val_data, step, name="val/"):
        with torch.no_grad():
            z_e_0 = self.base_dist.sample(self.cfg.trainer.nb_sample_partition_estimate_val,)
            energy_base_dist = self.energy(z_e_0).flatten(1).sum(1)
            log_partition_estimate = torch.logsumexp(-energy_base_dist,0) - math.log(energy_base_dist.shape[0])
            log(step, {"log_z":log_partition_estimate.item()}, logger=self.logger, name=name)
            log(step, {"energy_base_dist":energy_base_dist.mean().item()}, logger=self.logger, name=name)

            k=0
            if name == "val/":
                max_len = len(val_data)/10
            else :
                max_len = len(val_data)
            while k* self.cfg.dataset.batch_size<max_len:
                x = val_data[k*self.cfg.dataset.batch_size:(k+1)*self.cfg.dataset.batch_size].to(self.cfg.trainer.device)
                x_expanded = x.unsqueeze(0).expand(self.cfg.trainer.multiple_sample_val,x.shape[0],self.cfg.dataset.nc,self.cfg.dataset.img_size,self.cfg.dataset.img_size).flatten(0,1)
                expanded_batch_size = x.shape[0]*self.cfg.trainer.multiple_sample_val
                dic = {}
                mu_q, log_var_q = self.encoder(x_expanded).chunk(2,1)
                std_q = torch.exp(0.5*log_var_q)

                # Reparam trick
                eps = torch.randn_like(mu_q)
                z_q = (eps.mul(std_q).add_(mu_q)).reshape(expanded_batch_size, self.cfg.trainer.nz,)
                x_hat = self.generator(z_q)

                # Reconstruction loss :
                loss_g = self.generator.get_loss(x_hat, x_expanded).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0]).mean(dim=0)

                # KL without ebm
                KL_loss = 0.5 * (self.log_var_p- log_var_q- 1+ (log_var_q.exp() + mu_q.pow(2)) / self.log_var_p.exp())
                KL_loss = KL_loss.sum(dim=1).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0]).mean(dim=0)

                # Entropy posterior
                entropy_posterior = torch.sum(0.5 * (math.log(2 * math.pi) + log_var_q + 1), dim=1).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0]).mean(dim=0)

                # Gaussian mixture
                log_prob_mixture = self.prior.log_prob(z_q)

                log_prob_mixture = log_prob_mixture.reshape(self.cfg.trainer.multiple_sample_val,x.shape[0]).mean(dim=0)
               
                # Energy :
                energy_approximate = self.energy(z_q).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0])
                base_dist_z_approximate = self.base_dist.log_prob(z_q).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0])


                loss_ebm = (energy_approximate + log_partition_estimate.exp() - 1).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0]).mean(dim=0)
                loss_total = loss_g + KL_loss + loss_ebm
                elbo_mixture = -loss_g + entropy_posterior + log_prob_mixture

                dic_loss = {
                    "loss_g": loss_g,
                    "entropy_posterior": entropy_posterior,
                    "loss_ebm": loss_ebm,
                    "base_dist_z_approximate": base_dist_z_approximate.mean(dim=0),
                    "KL_loss_no_ebm": KL_loss,
                    "energy_approximate": energy_approximate.mean(dim=0),
                    "approx_elbo": -loss_total,
                    "elbo_no_ebm": -loss_g - KL_loss,
                    "elbo_mixture": elbo_mixture,
                    "mu_q": mu_q.flatten(1).mean(1).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0]).mean(dim=0),
                    "log_var_q": log_var_q.flatten(1).sum(1).reshape(self.cfg.trainer.multiple_sample_val,x.shape[0]).mean(dim=0),
                }
                for key, value in dic_loss.items():
                    if key not in dic:
                        dic[key] = []
                    dic[key].append(value.reshape(x.shape[0]))
                k = k + 1
            for key in dic:
                dic[key] = torch.stack(dic[key], dim=0).mean().item()
            log(step, dic, logger=self.logger, name=name)



    def draw_samples(self, x, step):
        batch_save = min(64, x.shape[0])
        z_e_0, z_g_0 = self.base_dist.sample(batch_save), self.base_dist.sample(batch_save)
    
        z_e_k = self.sampler_prior(z_e_0, self.energy, self.base_dist,)

        z_g_k = self.sampler_posterior(z_g_0,x[:batch_save], self.generator, self.energy, self.base_dist,)

        x_base, mu_base =self.generator.sample(z_e_0, return_mean=True)
        draw(x_base.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleBaseDistribution")
        draw(mu_base.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanBaseDistribution")
    
        x_prior, mu_prior =self.generator.sample(z_e_k, return_mean=True)
        draw(x_prior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleEBMPrior")
        draw(mu_prior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanEBMPrior")
 
        x_posterior, mu_posterior =self.generator.sample(z_g_k, return_mean=True)
        draw(x_posterior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleEBMPosterior")
        draw(mu_posterior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanEBMPosterior")

        x_reconstruction, mu_reconstruction =self.generator.sample(self.encoder(x[:batch_save]).chunk(2, 1)[0].reshape(-1, self.cfg.trainer.nz, 1, 1), return_mean=True)
        draw(x_reconstruction.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleReconstruction")
        draw(mu_reconstruction.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanReconstruction")
       
        extra_prior_samples = self.prior.sample(batch_save).reshape(batch_save, self.cfg.trainer.nz, 1, 1).to(self.cfg.trainer.device)
        x_prior, mu_prior = self.generator.sample(extra_prior_samples, return_mean=True)
        draw(x_prior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleExtraPrior")
        draw(mu_prior.reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanExtraPrior")


    def plot_latent(self, step):
        if self.cfg.trainer.nz != 2:
            pass
        else :
            data = self.sampler.sample_p_data(1000).to(self.cfg.trainer.device)

            len_samples = min(1000, data.shape[0])
            mu_q, log_var_q = self.encoder(data[:len_samples]).chunk(2,1)
            z_e_0, z_g_0 = self.base_dist.sample(len_samples), self.base_dist.sample(len_samples)
            z_e_k = self.sampler_prior(z_e_0, self.energy, self.base_dist,)
            z_g_k = self.sampler_posterior(z_g_0, data[:len_samples], self.generator, self.energy, self.base_dist,)



            energy_list_small_scale, energy_list_names, x, y = self.get_all_energies(z_e_0, min_x=-3, max_x=3)
            samples_aux = self.cut_samples(z_e_0, min_x=-3, max_x=3)
            plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Base Distribution SC")
            
            samples_aux = self.cut_samples(z_e_k, min_x=-3, max_x=3)
            plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Prior SC")

            samples_aux = self.cut_samples(mu_q, min_x=-3, max_x=3)
            plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Approximate Posterior SC")

            samples_aux = self.cut_samples(z_g_k, min_x=-3, max_x=3)
            plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Posterior SC")


            energy_list_large_scale, energy_list_names, x, y = self.get_all_energies(z_e_0, min_x=-10, max_x=10)
            samples_aux = self.cut_samples(z_e_0, min_x=-10, max_x=10)
            plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Base Distribution LC")
            
            samples_aux = self.cut_samples(z_e_k, min_x=-10, max_x=10)
            plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Prior LC")

            samples_aux = self.cut_samples(mu_q, min_x=-10, max_x=10)
            plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Approximate Posterior LC")

            samples_aux = self.cut_samples(z_g_k, min_x=-10, max_x=10)
            plot_contour(samples_aux, energy_list_large_scale, energy_list_names, x, y, step=step, logger=self.logger, title="Latent Posterior LC")


    def cut_samples(self, samples, min_x=-10, max_x =-10):
        min_y = min_x
        max_y = max_x
        tensor_min = torch.cat([torch.full_like(samples[:,0,None], min_x),torch.full_like(samples[:,1, None], min_y)], dim=1)
        tensor_max = torch.cat([torch.full_like(samples[:,0,None], max_x),torch.full_like(samples[:,1, None], max_y)], dim=1)
        samples = torch.where(samples < tensor_min, tensor_min, samples)
        samples = torch.where(samples > tensor_max, tensor_max, samples)
        return samples


    def get_all_energies(self, samples, min_x=-10, max_x=-10):
        samples_mean = samples.mean(0)
        samples_std = samples.std(0)
        # min_x = min(-3, samples_mean[0].min().item() - 3*samples_std[0].item())
        # max_x = max(3, samples_mean[0].max().item()+ 3*samples_std[0].item())
        # min_y = min(-3, samples_mean[1].min().item() - 3*samples_std[1].item())
        # max_y = max(3, samples_mean[1].max().item()+ 3*samples_std[1].item())
        # min_x = -10
        # max_x = 10
        min_y = min_x
        max_y = max_x
        
        grid_coarseness = self.cfg.trainer.grid_coarseness
        device = self.cfg.trainer.device


        x = np.linspace(min_x, max_x, grid_coarseness)
        y = np.linspace(min_y, max_y, grid_coarseness)
        xx, yy = np.meshgrid(x, y)
        xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
        xy = torch.from_numpy(xy).float().to(device)

        energy_base_dist = - self.base_dist.log_prob(xy).reshape(grid_coarseness,grid_coarseness,)
        energy_extra_prior = - self.prior.log_prob(xy).reshape(grid_coarseness,grid_coarseness)
        just_energy = self.energy(xy).reshape(grid_coarseness, grid_coarseness)
        energy_prior = just_energy + energy_base_dist

        energy_list = [energy_base_dist, energy_prior, energy_extra_prior, just_energy]
        energy_list_names = ["Base Distribution", "EBM Prior", "Extra Prior", "Just EBM"]

        return energy_list, energy_list_names, x, y