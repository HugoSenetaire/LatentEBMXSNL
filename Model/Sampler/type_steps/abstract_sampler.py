

import torch.nn as nn
from ..clamp_utils import clamp_all

class AbstractSamplerStep(nn.Module):
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

    def clamp_data(self, z_data):
        z_data = clamp_all(z_data, self.clip_grad_norm, self.clamp_min_grad, self.clamp_max_grad,)
        if self.hyperspherical:
            z_data = z_data / z_data.norm(dim=-1, keepdim=True)
        return z_data
    

    def clamp_grad(self, z_grad):
        z_grad = clamp_all(z_grad, self.clip_grad_norm, self.clamp_min_grad, self.clamp_max_grad,)
        return z_grad

    def to_save(self, i, z, samples = []):
        if i >= self.warmup_steps :
            real_step = i - self.warmup_steps
            if real_step % self.thinning == 0:
                samples.append(z.detach().clone())
        return samples
      

   