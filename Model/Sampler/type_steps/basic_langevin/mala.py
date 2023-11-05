
import torch.nn as nn
import torch
import torch.distributions as dist
from ..abstract_sampler import AbstractSamplerStep

def calculate_prop(z_init, z_step, grad_init, step_size):
    gamma = 0.5 * step_size * step_size
    val = torch.norm(z_init-gamma*grad_init-z_step)**2 / (4 * gamma)
    return val


class MALAPrior(AbstractSamplerStep):
    def __init__(self,
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
                hyperspherical = False):
        super().__init__(
            num_samples = num_samples,
            thinning = thinning,
            warmup_steps = warmup_steps,
            step_size = step_size,
            clamp_min_data = clamp_min_data,
            clamp_max_data = clamp_max_data,
            clamp_min_grad = clamp_min_grad,
            clamp_max_grad = clamp_max_grad,
            clip_data_norm = clip_data_norm,
            clip_grad_norm = clip_grad_norm,
            hyperspherical = hyperspherical,
        )


    
    def forward(self, z, energy, base_dist):
        z = z.clone().detach().requires_grad_(True)
        samples = []
        num_chains = z.shape[0]
        for i in range(self.nb_steps):
            en_init = energy(z).squeeze() - base_dist.log_prob(z).reshape(z.shape[0])
            z_grad_init = torch.autograd.grad(en_init.sum(), z)[0]
            z_grad_init = self.clamp_grad(z_grad_init)

            # z_mu_step = z - 0.5 * self.step_size * self.step_size * z_grad_init 
            z_step = z - 0.5 * self.step_size * self.step_size * z_grad_init + self.step_size * torch.randn_like(z)
            z_step = self.clamp_data(z_step)
            log_prob_forward = en_init + calculate_prop(z, z_step, z_grad_init, self.step_size)

            en_step = energy(z_step).squeeze() - base_dist.log_prob(z_step).reshape(z.shape[0])
            z_grad_step = torch.autograd.grad(en_step.sum(), z_step)[0]
            z_grad_step = self.clamp_grad(z_grad_step)
            log_prob_backward = en_step + calculate_prop(z_step, z, z_grad_step, self.step_size)

            accept_prob = torch.exp(log_prob_backward - log_prob_forward)
            accept_prob = torch.clamp(accept_prob, 0, 1)
            accept = torch.rand_like(accept_prob) < accept_prob
            z.data = torch.where(accept.unsqueeze(-1), z_step.data, z.data)

            samples = self.to_save(i, z, samples)


        samples = torch.stack(samples)
        samples = samples.flatten(0,1)
        assert samples.shape[0] == num_chains * self.num_samples

        return samples


class MALAPosterior(AbstractSamplerStep):
    def __init__(self,
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
                hyperspherical = False):
        super().__init__(
            num_samples = num_samples,
            thinning = thinning,
            warmup_steps = warmup_steps,
            step_size = step_size,
            clamp_min_data= clamp_min_data,
            clamp_max_data= clamp_max_data,
            clamp_min_grad= clamp_min_grad,
            clamp_max_grad= clamp_max_grad,
            clip_data_norm= clip_data_norm,
            clip_grad_norm= clip_grad_norm,
            hyperspherical= hyperspherical,
        )
       

    def forward(self, z, x, generator, energy, base_dist):
        z = z.clone().detach().requires_grad_(True)
        num_chains = z.shape[0]
        samples = []
        for i in range(self.nb_steps):
            param = generator(z)
            g_log_lkhd = generator.get_loss(param, x).sum(dim=0)
            grad_g = torch.autograd.grad(g_log_lkhd, z, retain_graph=True)[0]
            en_init = energy(z).squeeze() - base_dist.log_prob(z).reshape(z.shape[0])
            grad_e = torch.autograd.grad(en_init.sum(), z)[0]
            total_grad = grad_g + grad_e
            total_grad = self.clamp_grad(total_grad)

            z_step = z - 0.5 * self.step_size * self.step_size * (total_grad) + self.step_size * torch.randn_like(z)
            z_step.data = self.clamp_data(z_step.data)

            param_step = generator(z_step)
            g_log_lkhd_step = generator.get_loss(param_step, x).sum(dim=0)
            grad_g_step = torch.autograd.grad(g_log_lkhd_step, z_step, retain_graph=True)[0]
            en_step = energy(z_step).squeeze() - base_dist.log_prob(z_step).reshape(z.shape[0])
            grad_e_step = torch.autograd.grad(en_step.sum(), z_step)[0]
            total_grad_step = grad_g_step + grad_e_step
            total_grad_step = self.clamp_grad(total_grad_step)

            log_prob_forward = g_log_lkhd + en_init + calculate_prop(z, z_step, total_grad, self.step_size)
            log_prob_backward = g_log_lkhd_step + en_step + calculate_prop(z_step, z, total_grad_step, self.step_size)

            accept_prob = torch.exp(log_prob_backward - log_prob_forward)
            accept_prob = torch.clamp(accept_prob, 0, 1)
            accept = torch.rand_like(accept_prob) < accept_prob
            z.data = torch.where(accept.unsqueeze(-1), z_step.data, z.data)
            
            samples = self.to_save(i, z, samples)
       
        samples = torch.stack(samples)
        samples = samples.flatten(0,1)
        assert samples.shape[0] == num_chains * self.num_samples

        return samples
    
    