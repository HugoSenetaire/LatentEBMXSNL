
import torch.nn as nn
import torch
from pyro.infer.mcmc import MCMC, NUTS


class NutsPrior(nn.Module):
    def __init__(
        self,
        num_samples,
        thinning,
        warmup_steps,
        step_size,
        clamp_min_data = None,
        clamp_max_data = None,
        clamp_min_grad = None,
        clamp_max_grad = None,
        clip_data_norm = None,
        clip_grad_norm = None,
        hyperspherical = False,
        multiprocess = 'None',
    ):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.thinning = thinning
        self.step_size = step_size
        self.nb_steps = (self.num_samples-1) * self.thinning + 1 + self.warmup_steps
        self.clamp_min_data = clamp_min_data
        self.clamp_max_data = clamp_max_data
        self.clamp_min_grad = clamp_min_grad
        self.clamp_max_grad = clamp_max_grad
        self.clip_data_norm = clip_data_norm
        self.clip_grad_norm = clip_grad_norm
        self.hyperspherical = hyperspherical
        self.multiprocess = multiprocess

    def energy(self, dic_input, energy_function, base_dist,):
        current_z = dic_input[0]
        if self.multiprocess == 'Standard' or self.multiprocess=="None":
            if len(current_z.shape) == 1:
                current_z = current_z.unsqueeze(0)
            energy =  energy_function(current_z).reshape(current_z.shape[0]) - base_dist.log_prob(current_z).reshape(current_z.shape[0])
            return energy.sum(dim=0)
        elif self.multiprocess == 'Cheating':
            current_z = current_z.flatten(0,1)
            return energy_function(current_z).sum(dim=0) - base_dist.log_prob(current_z).sum(dim=0)
        

    def forward(self, z, energy, base_dist):
        input_size = z.shape[1:]
        num_chains = z.shape[0]
        hmc_kernel = NUTS(
            potential_fn=lambda dic_input: self.energy(dic_input, energy, base_dist),
            adapt_step_size=True,
        )

        print(f"Running NUTS with {num_chains} chains and multiprocess = {self.multiprocess}")
        if self.multiprocess == 'None':
            # One step size per sample, normal but long
            samples = []
            for z_i in z:
                mcmc = MCMC(
                    hmc_kernel,
                    num_samples=self.num_samples * self.thinning,
                    warmup_steps=self.warmup_steps,
                    initial_params={0: z_i.unsqueeze(0)},
                    num_chains=1,
                )
                mcmc.run()
                samples.append(mcmc.get_samples()[0].clone().detach().reshape(self.num_samples, self.thinning, *input_size)[:, 0])
            samples = torch.cat(samples, dim=0).reshape(num_chains * self.num_samples, *input_size)
        elif self.multiprocess == 'Cheating':
            # One step size for the whole batch, kind of cheating
            samples = []
            mcmc = MCMC(
                hmc_kernel,
                num_samples=self.num_samples * self.thinning,
                warmup_steps=self.warmup_steps,
                initial_params={0: z.unsqueeze(0)},
                num_chains=1,
            )
            mcmc.run()
            samples = mcmc.get_samples()[0].clone().detach().reshape(self.num_samples, self.thinning, num_chains, *input_size)[:, 0].flatten(0, 1)
        elif self.multiprocess == 'Standard':
            # One step size per chain, requires pickle, more memory, quite annoying.
            mcmc = MCMC(
                hmc_kernel,
                num_samples=self.num_samples * self.thinning,
                warmup_steps=self.warmup_steps,
                initial_params={0: z},
                num_chains=num_chains,
            )
            mcmc.run()
            samples = mcmc.get_samples()[
                0
            ]  # 0 is because I have defined initial parameters as 0
            samples = samples.reshape(
                self.num_samples, self.thinning, num_chains, *input_size
            )[:, 0].flatten(0, 1)

        return samples



class NutsPosterior(nn.Module):
    def __init__(
        self,
        num_samples,
        thinning,
        warmup_steps,
        step_size,
        clamp_min_data = None,
        clamp_max_data = None,
        clamp_min_grad = None,
        clamp_max_grad = None,
        clip_data_norm = None,
        clip_grad_norm = None,
        hyperspherical = False,
        multiprocess = False,
    ):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.thinning = thinning
        self.step_size = step_size
        self.nb_steps = (self.num_samples-1) * self.thinning + 1 + self.warmup_steps
        self.clamp_min_data = clamp_min_data
        self.clamp_max_data = clamp_max_data
        self.clamp_min_grad = clamp_min_grad
        self.clamp_max_grad = clamp_max_grad
        self.clip_data_norm = clip_data_norm
        self.clip_grad_norm = clip_grad_norm
        self.hyperspherical = hyperspherical
        self.multiprocess = multiprocess

    def energy(self, dic_input, x, generator, energy, base_dist):
        z = dic_input[0]
        if self.multiprocess == "Standard" or self.multiprocess=="None":
            if len(z.shape) == 1:
                z = z.unsqueeze(0)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            assert len(x) == len(z)
            param = generator(z)
            g_log_lkhd = generator.get_loss(param, x).sum(dim=0)
            en = energy(z).reshape(z.shape[0]) - base_dist.log_prob(z).reshape(z.shape[0])
            g_log_lkhd = g_log_lkhd.reshape(z.shape[0])
            en = en.reshape(z.shape[0])
            return en + g_log_lkhd

        elif self.multiprocess == "Cheating":
            z = z.flatten(0,1)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            assert len(x) == len(z)
            param = generator(z)
            g_log_lkhd = generator.get_loss(param, x)
            en = energy(z).reshape(z.shape[0]) - base_dist.log_prob(z).reshape(z.shape[0])
            g_log_lkhd = g_log_lkhd.reshape(z.shape[0])
            en = en.reshape(z.shape[0])
            return (en + g_log_lkhd).sum(dim=0)
        

    def forward(self, z, x, generator, energy, base_dist):
        num_chains = z.shape[0]
        input_size = z.shape[1:]
       
        print(f"Running NUTS with {num_chains} chains and multiprocess = {self.multiprocess}")
        if self.multiprocess == 'None':
            samples = []

            # One step size per sample
            for z_i,x_i in zip(z,x):
                z_i = z_i
                x_i = x_i.unsqueeze(0)
                hmc_kernel = NUTS(
                    potential_fn=lambda dic_input: self.energy(dic_input, x_i, generator, energy, base_dist),
                    adapt_step_size=True,
                )
                mcmc = MCMC(
                    hmc_kernel,
                    num_samples=self.num_samples * self.thinning,
                    warmup_steps=self.warmup_steps,
                    initial_params={0: z_i.unsqueeze(0)},
                    num_chains=1,
                )
                mcmc.run()
                samples.append(mcmc.get_samples()[0].clone().detach().reshape(self.num_samples, self.thinning, *input_size)[:, 0])
            samples = torch.cat(samples, dim=0).reshape(num_chains * self.num_samples, *input_size)
        elif self.multiprocess == 'Cheating':
            # One step size for the whole batch
            samples = []
            assert len(x) == len(x)
            hmc_kernel = NUTS(
                    potential_fn=lambda dic_input: self.energy(dic_input, x, generator, energy, base_dist),
                    adapt_step_size=True,
                )
           # Some issues exists with pyro when multiprocessing, I am always using the same x
            mcmc = MCMC(
                hmc_kernel,
                num_samples=self.num_samples * self.thinning,
                warmup_steps=self.warmup_steps,
                initial_params={0: z.unsqueeze(0)},
                num_chains=1,
            )
            mcmc.run()
            samples =  mcmc.get_samples()[0].clone().detach().reshape(self.num_samples, self.thinning, num_chains, *input_size)[:, 0]
            samples = samples.flatten(0,1)
                
        elif self.multiprocess == 'Standard':
            hmc_kernel = NUTS(
                    potential_fn=lambda dic_input: self.energy(dic_input, x, generator, energy, base_dist),
                    adapt_step_size=True,
                )
            mcmc = MCMC(
                hmc_kernel,
                num_samples=self.num_samples * self.thinning,
                warmup_steps=self.warmup_steps,
                initial_params={0: z},
                num_chains=num_chains,
            )
            mcmc.run()
            samples = mcmc.get_samples()[
                0
            ]  # 0 is because I have defined initial parameters as 0
            samples = samples.reshape(
                self.num_samples, self.thinning, num_chains, *input_size
            )[:, 0].flatten(0, 1)

        return samples
