import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tfm
import torch.utils.data.dataloader

from .clamp_utils import clamp_all


class Sampler():
    def __init__(self, data) -> None:
        self.data = data
        if isinstance(data, torch.utils.data.dataloader.DataLoader):
            self.iterator = iter(data)
        else:
            self.iterator = None

    def sample_p_data(self, batch_size):
        if self.iterator is None:
            return self.data[t.LongTensor(batch_size).random_(0, self.data.size(0))].detach()
        else:
            try:
                return next(self.iterator)[0]
            except StopIteration:
                self.iterator = iter(self.data)
                return next(self.iterator)[0]
    

class SampleLangevinPrior(nn.Module):
    def __init__(self, K, a, clamp_min_data = None, clamp_max_data = None, clamp_min_grad = None, clamp_max_grad = None, clip_data_norm = None, clip_grad_norm = None):
        super().__init__()
        self.K = K
        self.a = a
        self.clamp_min_data = clamp_min_data
        self.clamp_max_data = clamp_max_data
        self.clamp_min_grad = clamp_min_grad
        self.clamp_max_grad = clamp_max_grad
        self.clip_data_norm = clip_data_norm
        self.clip_grad_norm = clip_grad_norm


    def actual_energy(self, z, energy, base_dist):
        en = energy(z) - base_dist.log_prob(z).reshape(z.shape[0])
        return en
    
    def forward(self, z, energy, base_dist):
        z = z.clone().detach().requires_grad_(True)
        for i in range(self.K):
            en = energy(z).squeeze() - base_dist.log_prob(z).reshape(z.shape[0])
            z_grad = torch.autograd.grad(en.sum(), z)[0]

            z_grad = clamp_all(z_grad, self.clip_grad_norm, self.clamp_min_grad, self.clamp_max_grad,)
            z_grad_norm = z_grad.view(z.shape[0], -1).norm(dim=1).mean(0)

            z.data = z.data - 0.5 * self.a * self.a * z_grad + self.a * torch.randn_like(z).data

            z.data = clamp_all(z.data, self.clamp_min_data, self.clamp_max_data, self.clip_data_norm)
        
        return z.detach(), z_grad_norm
    
class SampleLangevinPosterior(nn.Module):
    def __init__(self, K, a, clamp_min_data = None, clamp_max_data = None, clamp_min_grad = None, clamp_max_grad = None, clip_data_norm = None, clip_grad_norm = None):
        super().__init__()
        self.K = K
        self.a = a
        self.clamp_min_data = clamp_min_data
        self.clamp_max_data = clamp_max_data
        self.clamp_min_grad = clamp_min_grad
        self.clamp_max_grad = clamp_max_grad
        self.clip_data_norm = clip_data_norm
        self.clip_grad_norm = clip_grad_norm


    def actual_energy(self, z, energy, generator,):
        en = energy(z) + torch.norm(z, dim=1)
        return en
    
    def forward(self, z, x, generator, energy, base_dist):
        z = z.clone().detach().requires_grad_(True)
        for i in range(self.K):
            param = generator(z)
            g_log_lkhd = generator.get_loss(param, x).sum(dim=0)
            grad_g = t.autograd.grad(g_log_lkhd, z, retain_graph=True)[0]

            en = energy(z).squeeze() - base_dist.log_prob(z).reshape(z.shape[0])
            grad_e = t.autograd.grad(en.sum(), z)[0]

            total_grad = grad_g + grad_e
            clamp_all(total_grad, self.clip_grad_norm, self.clamp_min_grad, self.clamp_max_grad,)

            z.data = z.data - 0.5 * self.a * self.a * (total_grad) + self.a * t.randn_like(z).data

            z_grad_g_grad_norm = grad_g.view(x.shape[0], -1).norm(dim=1).mean()
            z_grad_e_grad_norm = grad_e.view(x.shape[0], -1).norm(dim=1).mean()

            z.data = clamp_all(z.data, self.clip_data_norm, self.clamp_min_data, self.clamp_max_data,)

        return z.detach(), z_grad_g_grad_norm, z_grad_e_grad_norm


