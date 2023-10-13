import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tfm
import torch.utils.data.dataloader


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
    def __init__(self, K, a, clamp_min=None, clamp_max=None, clip_grad_norm=None):
        super().__init__()
        self.K = K
        self.a = a
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.clip_grad_norm = clip_grad_norm

    def actual_energy(self, z, energy, base_dist):
        en = energy(z) - base_dist.log_prob(z).reshape(z.shape[0])
        return en
    
    def forward(self, z, energy, base_dist):
        z = z.clone().detach().requires_grad_(True)
        for i in range(self.K):
            en = energy(z).squeeze() - base_dist.log_prob(z).reshape(z.shape[0])
            z_grad = torch.autograd.grad(en.sum(), z)[0]

            if self.clip_grad_norm is not None:
                z_grad = nn.utils.clip_grad_norm_(z_grad, self.clip_grad_norm)
            
            z.data = z.data - 0.5 * self.a * self.a * z_grad + self.a * torch.randn_like(z).data
            z_grad_norm = z_grad.view(z.shape[0], -1).norm(dim=1).mean()

            if self.clamp_min is not None:
                z.data = z.data.clamp(min=self.clamp_min)
            if self.clamp_max is not None:
                z.data = z.data.clamp(max=self.clamp_max)
        
        return z.detach(), z_grad_norm
    
class SampleLangevinPosterior(nn.Module):
    def __init__(self, K, a, clamp_min=None, clamp_max=None, clip_grad_norm=None):
        super().__init__()
        self.K = K
        self.a = a
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.clip_grad_norm = clip_grad_norm

    def actual_energy(self, z, energy, generator,):
        en = energy(z)+ torch.norm(z, dim=1)
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
            if self.clip_grad_norm is not None:
                total_grad = nn.utils.clip_grad_norm_(total_grad, self.clip_grad_norm)

            z.data = z.data - 0.5 * self.a * self.a * (total_grad) + self.a * t.randn_like(z).data

            z_grad_g_grad_norm = grad_g.view(x.shape[0], -1).norm(dim=1).mean()
            z_grad_e_grad_norm = grad_e.view(x.shape[0], -1).norm(dim=1).mean()

            if self.clamp_min is not None:
                z.data = z.data.clamp(min=self.clamp_min)
            
            if self.clamp_max is not None:
                z.data = z.data.clamp(max=self.clamp_max)
        return z.detach(), z_grad_g_grad_norm, z_grad_e_grad_norm


