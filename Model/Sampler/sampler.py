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
    def __init__(self, K, a, trick=True):
        super().__init__()
        self.K = K
        self.a = a
        self.trick = trick

    def actual_energy(self, z, energy, base_dist):
        en = energy(z) - base_dist.log_prob(z).reshape(z.shape[0])
        return en
    
    def forward(self, z, energy, base_dist):
        z = z.clone().detach().requires_grad_(True)
        for i in range(self.K):
            en = energy(z).squeeze() - base_dist.log_prob(z).squeeze()
            z_grad = t.autograd.grad(en.sum(), z)[0]
            z.data = z.data - 0.5 * self.a * self.a * z_grad + self.a * t.randn_like(z).data

        return z.detach()
    
class SampleLangevinPosterior(nn.Module):
    def __init__(self, K, a,):
        super().__init__()
        self.K = K
        self.a = a

    def actual_energy(self, z, energy, generator,):
        en = energy(z)+ torch.norm(z, dim=1)
        return en
    
    def forward(self, z, x, generator, energy, base_dist):
        z = z.clone().detach().requires_grad_(True)
        for i in range(self.K):
            x_hat = generator(z)
            g_log_lkhd = generator.get_loss(x_hat, x).mean(dim=0)
            grad_g = t.autograd.grad(g_log_lkhd, z)[0]
            en = energy(z).squeeze() - base_dist.log_prob(z).reshape(z.shape[0])
            grad_e = t.autograd.grad(en.sum(), z)[0]
            z.data = z.data - 0.5 * self.a * self.a * (grad_g + grad_e) + self.a * t.randn_like(z).data
        return z.detach()


