import math
import torch
import torch.nn as nn
import tqdm
import time
import wandb
import os
import numpy as np
import re
import pickle 

from omegaconf import OmegaConf

from ..Generator import AbstractGenerator
from ..Encoder import AbstractEncoder
from ..Energy import get_energy_network
from ..Optim import get_optimizer, grad_clipping
from ..Optim.Schedulers import get_scheduler
from ..Prior import get_prior, get_extra_prior
from ..Sampler import get_posterior_sampler, get_prior_sampler
from ..ProposalDistribution import get_proposal
from ..Utils.log_utils import log, draw, get_extremum, plot_contour
from ..Utils.aggregate_posterior import AggregatePosterior
from ..Utils.utils_fid.utils import calculate_activation, calculate_frechet_distance
from ..Utils.utils_fid.inception_v3 import InceptionV3

class AbstractTrainer:
    def __init__(
        self,
        cfg,
        test = False,
        path_weights = None,
        load_iter=None,
    ) -> None:

        self.cfg = cfg

        self.generator = AbstractGenerator(cfg).to(cfg.trainer.device)
        self.energy = get_energy_network(cfg.energy.network_name, cfg.trainer.nz, cfg.energy.ndf)
        self.prior = get_prior(cfg.trainer.nz, cfg.prior).to(cfg.trainer.device)
        self.extra_prior = get_extra_prior(cfg.trainer.nz, cfg.extra_prior).to(cfg.trainer.device)
        self.proposal = get_proposal(cfg.trainer.nz, cfg.proposal)
        if self.proposal is not None :
            self.proposal = self.proposal.to(cfg.trainer.device)
        self.encoder = AbstractEncoder(cfg, cfg.trainer.nz, cfg.dataset.nc).to(cfg.trainer.device)

        self.sampler_prior = get_prior_sampler(cfg.sampler_prior)
        self.sampler_posterior = get_posterior_sampler(cfg.sampler_posterior)
        self.mse = nn.MSELoss(reduction="sum")
        self.mse_test = nn.MSELoss(reduction='none')
        self.log_var_p = torch.tensor(0, device=cfg.trainer.device, dtype=torch.float32)
        if cfg.trainer.log_dir is None:
            cfg.trainer.log_dir = os.path.join(cfg.machine.root, "logs",)
            print("Setting log dir to " + cfg.trainer.log_dir)
        

        if test :
            project = "Test_LatentEBM_{}_{}_v3".format(cfg.dataset.dataset_name,str(cfg.trainer.nz))
            name = path_weights.split("/")[-1]
            assert path_weights is not None, "You need to specify a path to load the weights"
            self.path_weights = path_weights
            self.load_model(path_weights, load_iter=load_iter)
        else :
            project = "LatentEBM_{}_{}_v3".format(cfg.dataset.dataset_name,str(cfg.trainer.nz))
            name = cfg.trainer.trainer_name + "_" + cfg.prior.prior_name + "_" + cfg.encoder.latent_distribution_name + time.strftime("%Y%m%d-%H%M%S")
        
        self.save_dir = os.path.join(os.path.join(cfg.trainer.log_dir, project), name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        pickle.dump(cfg, open(os.path.join(self.save_dir, "cfg.pkl"), "wb"))
        self.logger = wandb.init(
            project=project,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.trainer.log_dir,
            name= name,
        )
        self.n_iter = cfg.trainer.n_iter
        self.n_iter_pretrain = cfg.trainer.n_iter_pretrain
        self.n_iter_pretrain_encoder = cfg.trainer.n_iter_pretrain_encoder
        self.compile()
        
    def get_fixed_x(self, train_dataloader, val_dataloader, test_dataloader):
        max_batch = max(self.cfg.sampler_posterior.num_chains_test, 256)
        if test_dataloader is not None :
            current_data_loader = test_dataloader
        elif val_dataloader is not None:
            current_data_loader = val_dataloader
        else :
            current_data_loader = train_dataloader
        x = []
        while (len(x)<max_batch):
            x.append(next(iter(current_data_loader))[0].to(self.cfg.trainer.device))
        x = torch.cat(x, dim=0)[:max_batch]
        self.x_fixed = x



    def save_model(self, name=""):
        torch.save(self.generator.state_dict(), os.path.join(self.save_dir, "generator_{}.pt".format(name)))
        torch.save(self.encoder.state_dict(), os.path.join(self.save_dir, "encoder_{}.pt".format(name)))
        torch.save(self.energy.state_dict(), os.path.join(self.save_dir, "energy_{}.pt".format(name)))
        torch.save(self.extra_prior.state_dict(), os.path.join(self.save_dir, "extra_prior_{}.pt".format(name)))

    def load_model(self, path_weights, load_iter=None):
        set_index = set()
        for path in os.listdir(path_weights):
            index = re.findall('^generator_(\d+).pt$', path)
            if len(index) > 0:
                set_index.update(index)
        if load_iter is None:
            load_iter = max(set_index)
        else :
            assert load_iter in set_index, "The index {} is not in the set of index {}".format(load_iter, set_index)
        
        self.generator.load_state_dict(torch.load(os.path.join(path_weights, "generator_{}.pt".format(load_iter))))
        self.encoder.load_state_dict(torch.load(os.path.join(path_weights, "encoder_{}.pt".format(load_iter))))
        self.energy.load_state_dict(torch.load(os.path.join(path_weights, "energy_{}.pt".format(load_iter))))
        self.extra_prior.load_state_dict(torch.load(os.path.join(path_weights, "extra_prior_{}.pt".format(load_iter))))
        self.compile()

    def compile(self):
        self.generator.to(self.cfg.trainer.device)
        self.opt_generator = get_optimizer(self.cfg.optim_generator, self.generator)
        self.sch_generator = get_scheduler(self.cfg.scheduler_generator, self.opt_generator)

        self.encoder.to(self.cfg.trainer.device)
        self.opt_encoder = get_optimizer(self.cfg.optim_encoder, self.encoder)
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
        self.get_fixed_x(train_dataloader, val_dataloader, test_dataloader)
        for self.global_step in tqdm.tqdm(range(self.n_iter_pretrain_encoder + self.n_iter_pretrain + self.n_iter)):

            # Eval
            if (self.global_step) % self.cfg.trainer.val_every == 0 and val_dataloader is not None:
                self.eval(val_dataloader, self.global_step)
                self.fid_eval(val_data=test_dataloader, step=self.global_step, name="val/")
                
            # Test
            if (self.global_step)%self.cfg.trainer.test_every == 0 and test_dataloader is not None and self.global_step>1 :
                self.eval(test_dataloader, self.global_step, name="test/")
                self.fid_eval(val_data=test_dataloader, step=self.global_step, name="test/")

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

            # Save images
            if self.global_step % self.cfg.trainer.save_images_every == 0 :
                self.draw_samples(x, self.global_step)
                self.plot_latent(dataloader=train_dataloader,step = self.global_step)

            if self.global_step < self.n_iter_pretrain_encoder:
                dic_loss = self.train_step_only_encoder(x, self.global_step)
            elif self.global_step < self.n_iter_pretrain + self.n_iter_pretrain_encoder:
                dic_loss = self.train_step_standard_elbo(x, self.global_step)
            else:
                dic_loss = self.train_step(x, self.global_step)

            

            # Log
            if self.global_step % self.cfg.trainer.log_every == 0:
                log(self.global_step, dic_loss, logger=self.logger)

           

            # Save models :
            if self.global_step % self.cfg.trainer.save_every == 0 and self.global_step>1:
                self.save_model(name=str(self.global_step))



            

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
            log_partition_estimate = None
            total_energy = 0
            nb_sample_partition = getattr(self.cfg.trainer, "nb_sample_partition_estimate_{}".format(name[:-1]))
            for k in range(int(np.ceil(nb_sample_partition/batch_size_val))):
                if self.proposal is not None:
                    z_e_0 = self.proposal.sample(nb_sample_partition,)[:nb_sample_partition-sampled]
                else :
                    z_e_0 = self.prior.sample(nb_sample_partition,)[:nb_sample_partition-sampled]
                sampled += z_e_0.shape[0]
                current_energy = self.energy(z_e_0).flatten(1).sum(1)
                total_energy += self.energy(z_e_0).sum(0)
                if self.proposal is not None:
                    current_energy = current_energy + self.proposal.log_prob(z_e_0).reshape(z_e_0.shape[0]) - self.prior.log_prob(z_e_0).reshape(z_e_0.shape[0])

                if log_partition_estimate is None :
                    log_partition_estimate = torch.logsumexp(-current_energy,0)
                else :
                    to_sum = torch.cat([log_partition_estimate.unsqueeze(0), -current_energy], dim=0)
                    log_partition_estimate = torch.logsumexp(to_sum,0)
            log_partition_estimate = log_partition_estimate - math.log(sampled)
            log(step, {"log_z":log_partition_estimate.item()}, logger=self.logger, name=name)
            if self.proposal is not None :
                log(step, {"energy_proposal": (total_energy/sampled).item()}, logger=self.logger, name=name)
            else :
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
        batch_save = min(self.cfg.dataset.batch_size_val, val_data.dataset.__len__())
        nb_sample = getattr(self.cfg.trainer, "nb_sample_fid_{}".format(name[:-1]))
        nb_batch = int(np.ceil(nb_sample/batch_save))

        inception = InceptionV3([3])
        try :
            inception = inception.to(self.cfg.trainer.device)
            current_device = self.cfg.trainer.device
        except RuntimeError as e:
            print(e)
            print("HERE")
            inception = inception.to("cpu")
            current_device = "cpu"
            
        print("Calculating Inception Score using {} images and device {}".format(nb_sample, current_device))
        # Statistics for generated :
        ranger = tqdm.tqdm(range(nb_batch), desc=f"fid_{name[:-1]}_gen", position=1, leave=False)
        activation_sample = []
        activation_mean = []
        activation_prior_sample = []
        activation_prior_mean = []
        for i in ranger:
            if self.cfg.sampler_prior.sampler_name == "nuts":
                z_e_0, z_e_k = self.handle_specific_sampler_prior(batch_save)
            else :
                if self.sampler_prior.num_samples >1:
                    z_e_0 = self.prior.sample(batch_save)
                    self.sampler_prior.num_samples = int(batch_save/8)
                    z_e_k = self.sampler_prior(z_e_0[:8], self.energy, self.prior,)
                    self.sampler_prior.num_samples = self.cfg.sampler_prior.num_samples
                else :
                    z_e_0 = self.prior.sample(batch_save)
                    z_e_k = self.sampler_prior(z_e_0, self.energy, self.prior,)
                
            x_prior_sample, x_prior_mean = self.generator.sample(z_e_0, return_mean=True)
            x_sample, x_mean = self.generator.sample(z_e_k, return_mean=True)
            with torch.no_grad():
                x_sample = x_sample.detach().to(current_device).reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)
                x_mean = x_mean.detach().to(current_device).reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)
                x_sample = x_sample.expand(batch_save, 3, self.cfg.dataset.img_size, self.cfg.dataset.img_size)
                x_mean = x_mean.expand(batch_save, 3, self.cfg.dataset.img_size, self.cfg.dataset.img_size)
                x_prior_sample = x_prior_sample.detach().to(current_device).reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)
                x_prior_mean = x_prior_mean.detach().to(current_device).reshape(batch_save, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)
                x_prior_sample = x_prior_sample.expand(batch_save, 3, self.cfg.dataset.img_size, self.cfg.dataset.img_size)
                x_prior_mean = x_prior_mean.expand(batch_save, 3, self.cfg.dataset.img_size, self.cfg.dataset.img_size)
                activation_prior_sample.append(calculate_activation(x_prior_sample, inception, device=current_device))
                activation_prior_mean.append(calculate_activation(x_prior_mean, inception, device=current_device))
                activation_sample.append(calculate_activation(x_sample, inception, device=current_device))
                activation_mean.append(calculate_activation(x_mean, inception, device=current_device))
        activation_sample = np.concatenate(activation_sample, axis=0)
        activation_mean = np.concatenate(activation_mean, axis=0)
        activation_prior_sample = np.concatenate(activation_prior_sample, axis=0)
        activation_prior_mean = np.concatenate(activation_prior_mean, axis=0)
        mu_sample = np.mean(activation_sample, axis=0)
        sigma_sample = np.cov(activation_sample, rowvar=False)
        mu_mean = np.mean(activation_mean, axis=0)
        sigma_mean = np.cov(activation_mean, rowvar=False)
        mu_prior_sample = np.mean(activation_prior_sample, axis=0)
        sigma_prior_sample = np.cov(activation_prior_sample, rowvar=False)
        mu_prior_mean = np.mean(activation_prior_mean, axis=0)
        sigma_prior_mean = np.cov(activation_prior_mean, rowvar=False)

        
        activation_data = []
        ranger = tqdm.tqdm(iter(val_data), desc=f"fid_{name[:-1]}_data", position=1, leave=False)
        for batch in ranger:
            with torch.no_grad():
                data = batch[0].to(current_device).reshape(batch[0].shape[0], self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size)
                data = data.expand(batch[0].shape[0], 3, self.cfg.dataset.img_size, self.cfg.dataset.img_size)
                activation_data.append(calculate_activation(data, inception, device=current_device))
        activation_data = np.concatenate(activation_data, axis=0)
        mu_data = np.mean(activation_data, axis=0)
        sigma_data = np.cov(activation_data, rowvar=False)

        fid_sample = calculate_frechet_distance(mu_data, sigma_data, mu_sample, sigma_sample)
        fid_mean = calculate_frechet_distance(mu_data, sigma_data, mu_mean, sigma_mean)
        fid_prior_sample = calculate_frechet_distance(mu_data, sigma_data, mu_prior_sample, sigma_prior_sample)
        fid_prior_mean = calculate_frechet_distance(mu_data, sigma_data, mu_prior_mean, sigma_prior_mean)


        log(step, {"fid":fid_sample}, logger=self.logger, name=name)
        log(step, {"fid_mean":fid_mean}, logger=self.logger, name=name)
        log(step, {"fid_prior_sample":fid_prior_sample}, logger=self.logger, name=name)
        log(step, {"fid_prior_mean":fid_prior_mean}, logger=self.logger, name=name)

    def draw_samples(self, x, step):
        print("Drawing samples")
        batch_save_bast_dist = self.cfg.sampler_prior.num_chains_test
        batch_save_prior = self.cfg.sampler_prior.num_chains_test * self.cfg.sampler_prior.num_samples
        batch_save_posterior = self.cfg.sampler_posterior.num_chains_test * self.cfg.sampler_posterior.num_samples
        z_e_0, z_g_0 = self.prior.sample(batch_save_bast_dist), self.prior.sample(batch_save_bast_dist)
        z_e_k = self.sampler_prior(z_e_0[:self.cfg.sampler_prior.num_chains_test],
                                    self.energy,
                                    self.prior,)
        z_g_k = self.sampler_posterior(z_g_0[:self.cfg.sampler_posterior.num_chains_test],
                                    self.x_fixed[:self.cfg.sampler_posterior.num_chains_test],
                                    self.generator,
                                    self.energy,
                                    self.prior,)
        # z_e_k, z_grad_norm = self.sampler_prior(z_e_0, self.energy, self.prior,)
        # z_g_k, z_g_grad_norm, z_e_grad_norm = self.sampler_posterior(z_g_0,x[:batch_save], self.generator, self.energy, self.prior,)

        with torch.no_grad():
            x_base, mu_base =self.generator.sample(z_e_0, return_mean=True)
            draw(x_base.reshape(batch_save_bast_dist, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleBaseDistribution")
            draw(mu_base.reshape(batch_save_bast_dist, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanBaseDistribution")
        
            x_prior, mu_prior =self.generator.sample(z_e_k, return_mean=True)
            draw(x_prior.reshape(batch_save_prior, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleEBMPrior")
            draw(mu_prior.reshape(batch_save_prior, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanEBMPrior")
    
            x_posterior, mu_posterior = self.generator.sample(z_g_k, return_mean=True)
            draw(x_posterior.reshape(batch_save_posterior, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleEBMPosterior")
            draw(mu_posterior.reshape(batch_save_posterior, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanEBMPosterior")


            param = self.encoder(self.x_fixed[:batch_save_posterior])
            sample_mean, sample_std = self.encoder.latent_distribution.get_plots(param,)

            x_reconstruction, mu_reconstruction = self.generator.sample(sample_mean, return_mean=True)
            draw(x_reconstruction.reshape(batch_save_posterior, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleReconstruction")
            draw(mu_reconstruction.reshape(batch_save_posterior, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanReconstruction")
        
            extra_prior_samples = self.extra_prior.sample(batch_save_prior).reshape(batch_save_prior, self.cfg.trainer.nz, 1, 1).to(self.cfg.trainer.device)
            x_prior, mu_prior = self.generator.sample(extra_prior_samples, return_mean=True)
            draw(x_prior.reshape(batch_save_posterior, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="SampleExtraPrior")
            draw(mu_prior.reshape(batch_save_posterior, self.cfg.dataset.nc, self.cfg.dataset.img_size, self.cfg.dataset.img_size), step, self.logger, transform_back_name=self.cfg.dataset.transform_back_name, aux_name="MeanExtraPrior")

    def plot_latent(self, dataloader, step):
        if self.cfg.trainer.nz != 2:
            pass
        else :
            print("Plotting latent in 2D to see the distribution")
            iterator = iter(dataloader)
            batch = next(iterator)
            data = batch[0].to(self.cfg.trainer.device)
            targets = batch[1].to(self.cfg.trainer.device)
            while len(data)<5000:
                batch = next(iterator)
                data = torch.cat([data,batch[0].to(self.cfg.trainer.device)], dim=0)
                targets = torch.cat([targets, batch[1].to(self.cfg.trainer.device)], dim=0)
            data = data[:5000]
            targets = targets[:5000]
            len_samples = min(5000, data.shape[0])
            params = self.encoder(data[:len_samples])
            # no_treatment_params = params[:,:2]
            mu_q, log_var_q = self.encoder.latent_distribution.get_plots(params)
            samples_approx_post = self.encoder.latent_distribution.r_sample(params, n_samples=10).reshape(10*len_samples, self.cfg.trainer.nz)

            if self.cfg.sampler_prior.sampler_name == "nuts":
                z_e_0, z_e_k = self.handle_specific_sampler_prior(nb_sample=1000)
            else :
                z_e_0 = self.prior.sample(len_samples)
                z_e_k = self.sampler_prior(z_e_0[:int(len_samples/self.sampler_prior.num_samples)], self.energy, self.prior,)

            if self.cfg.sampler_prior.sampler_name == "nuts":
                z_g_0, z_g_k = self.handle_specific_sampler_posterior(data,nb_sample=1000)
            else :
                z_g_0 = self.prior.sample(len_samples)
                limit = int(len_samples/self.sampler_posterior.num_samples)
                z_g_k = self.sampler_posterior(z_g_0[:limit], data[:limit], self.generator, self.energy, self.prior,)
    
         
            liste_samples = [z_e_0, z_e_k, mu_q, samples_approx_post, z_g_k]
            liste_samples_name = ["Latent Base Distribution", "Latent Prior", "Latent Approximate Posterior Mu ", "Latent Approximate Posterior Sample", "Latent Posterior"]

            # liste_samples.append(no_treatment_params)
            # liste_samples_name.append("Latent Approximate Posterior Mu No Treatment")

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
                self.plot_samples_2d(liste_samples, -3, 3, liste_samples_name, step, params=params, params_reverse=params_reverse, targets = targets)
                self.plot_samples_2d(liste_samples, -10, 10, liste_samples_name, step, params=params, params_reverse=params_reverse, targets = targets)
                self.plot_samples_2d(liste_samples, -30, 30, liste_samples_name, step, params=params, params_reverse=params_reverse, targets = targets)
            elif "hyperspherical" in self.cfg.prior.prior_name:
                self.plot_samples_2d(liste_samples, -2, 2, liste_samples_name, step, params=params, params_reverse=params_reverse, targets = targets)
            elif self.cfg.prior.prior_name =='uniform' :
                self.plot_samples_2d(liste_samples, self.cfg.prior.min+1e-2, self.cfg.prior.max-1e-2, liste_samples_name, step, params=params, params_reverse=params_reverse, targets = targets)
            else :
                raise ValueError("Prior name not recognized")

    def plot_samples_2d(self, samples, min_x, max_x, liste_samples_name, step, params = None, params_reverse=None, targets = None):
        energy_list_small_scale, energy_list_names, x, y = self.get_all_energies(samples[0], min_x=min_x, max_x=max_x, params = params, params_reverse = params_reverse)
        for sample, samples_name in zip(samples, liste_samples_name):
                
            samples_aux = self.cut_samples(sample, min_x=min_x, max_x=max_x)
            title = samples_name+f" [{str(min_x)}, {str(max_x)}]"
            if targets is not None and "posterior" in samples_name.lower():
                plot_contour(samples_aux, energy_list_small_scale, energy_list_names, x, y, step=step, title=title, logger=self.logger, targets=targets)
            else :
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
        if params is not None and self.cfg.encoder.latent_distribution_name != 'uniform' and "mises" not in self.cfg.encoder.latent_distribution_name and "symmetrical" not in self.cfg.encoder.latent_distribution_name :
            dic_params, _ = self.encoder.latent_distribution.get_params(params)
            dist_posterior = self.encoder.latent_distribution.get_distribution(params, dic_params=dic_params)
            aggregate = AggregatePosterior(dist_posterior, params.shape[0], device = device)
            aggregate_energy = -aggregate.log_prob(xy).reshape(grid_coarseness, grid_coarseness)
            energy_list.append(aggregate_energy)
            energy_list_names.append("Aggregate Posterior")
            
        if params_reverse is not None and self.cfg.encoder.latent_distribution_name != 'uniform' and "mises" not in self.cfg.encoder.latent_distribution_name and "symmetrical" not in self.cfg.encoder.latent_distribution_name:
        # if params_reverse is not None and self.cfg.encoder.latent_distribution_name == 'gaussian':
            dic_params, _ = self.reverse_encoder.latent_distribution.get_params(params)
            dist_posterior = self.reverse_encoder.latent_distribution.get_distribution(params, dic_params=dic_params)
            aggregate = AggregatePosterior(dist_posterior, params.shape[0], device = device)
            aggregate_energy = -aggregate.log_prob(xy).reshape(grid_coarseness, grid_coarseness)
            energy_list.append(aggregate_energy)
            energy_list_names.append("Aggregate Posterior Reverse")

        return energy_list, energy_list_names, x, y
    


    def handle_specific_sampler_prior(self, nb_sample = 5000):
        self.sampler_prior.multiprocess = "Cheating"
        self.sampler_prior.num_samples = 1
        z_e_0 = self.prior.sample(nb_sample)
        z_e_k = self.sampler_prior(z_e_0, self.energy, self.prior,)
        self.sampler_prior.multiprocess = self.cfg.sampler_prior.multiprocess
        self.sampler_prior.num_samples = self.cfg.sampler_prior.num_samples
        return z_e_0, z_e_k

    def handle_specific_sampler_posterior(self, data, nb_sample = 5000):
        self.sampler_posterior.multiprocess = "Cheating"
        self.sampler_posterior.num_samples = 1
        nb_chain = 20
        batch_per_chain = int(np.ceil(nb_sample/nb_chain))
        z_g_0 = self.prior.sample(nb_sample)
        z_g_k = []
        for k in range(nb_chain):
            current_data = data[k*batch_per_chain:(k+1)*batch_per_chain]
            current_z_g_0 = z_g_0[k*batch_per_chain:(k+1)*batch_per_chain]
            z_g_k.append(self.sampler_posterior(current_z_g_0, current_data, generator = self.generator, energy = self.energy, base_dist = self.prior,))
        z_g_k = torch.cat(z_g_k, dim=0)
        self.sampler_posterior.multiprocess = self.cfg.sampler_posterior.multiprocess
        self.sampler_posterior.num_samples = self.cfg.sampler_posterior.num_samples
        return z_g_0, z_g_k