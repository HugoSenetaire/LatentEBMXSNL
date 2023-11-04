
import torch.nn as nn
import torch
from ..abstract_sampler import AbstractSamplerStep



class LangevinPrior(AbstractSamplerStep):
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
            en = energy(z).squeeze() - base_dist.log_prob(z).reshape(z.shape[0])

            z_grad = torch.autograd.grad(en.sum(), z)[0]
            z_grad = self.clamp_grad(z_grad)

            z.data = z.data - 0.5 * self.step_size * self.step_size * z_grad + self.step_size * torch.randn_like(z).data
            z.data = self.clamp_data(z.data)

            samples = self.to_save(i, z, samples)


        samples = torch.stack(samples)
        samples = samples.flatten(0,1)
        assert samples.shape[0] == num_chains * self.num_samples

        return samples


class LangevinPosterior(AbstractSamplerStep):
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

            en = energy(z).squeeze() - base_dist.log_prob(z).reshape(z.shape[0])
            grad_e = torch.autograd.grad(en.sum(), z)[0]

            total_grad = grad_g + grad_e
            total_grad = self.clamp_grad(total_grad)

            z.data = z.data - 0.5 * self.step_size * self.step_size * (total_grad) + self.step_size * torch.randn_like(z).data
            z.data = self.clamp_data(z.data)

            samples = self.to_save(i, z, samples)

       
        samples = torch.stack(samples)
        samples = samples.flatten(0,1)
        assert samples.shape[0] == num_chains * self.num_samples

        return samples