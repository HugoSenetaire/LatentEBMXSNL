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
    


def sample_p_0(n, nz, device):
    return t.randn(*[n, nz, 1, 1]).to(device)


def sample_langevin_prior(z, E, K_0, a_0):
    z = z.clone().detach().requires_grad_(True)
    for i in range(K_0):
        en = E(z)
        z_grad = t.autograd.grad(en.sum(), z)[0]
        z.data = z.data - 0.5 * a_0 * a_0 * (z_grad + 1.0 / z.data) + a_0 * t.randn_like(z).data
    return z.detach()

def sample_langevin_posterior(z, x, G, E, K_1, a_1, loss_reconstruction ):
    z = z.clone().detach().requires_grad_(True)
    for i in range(K_1):
        x_hat = G(z)
        # g_log_lkhd = 1.0 / (2.0 * llhd_sigma * llhd_sigma) * mse(x_hat, x)
        g_log_lkhd = loss_reconstruction(x_hat, x).mean(dim=0)
        grad_g = t.autograd.grad(g_log_lkhd, z)[0]
        en = E(z)
        grad_e = t.autograd.grad(en.sum(), z)[0]
        z.data = z.data - 0.5 * a_1 * a_1 * (grad_g + grad_e + 1.0 / z.data) + a_1 * t.randn_like(z).data
    return z.detach()



def sample_langevin_prior_notrick(z, E, K_0, a_0):
    z = z.clone().detach().requires_grad_(True)
    for i in range(K_0):
        en = E(z)
        z_grad = t.autograd.grad(en.sum(), z)[0]
        z.data = z.data - 0.5 * a_0 * a_0 * (z_grad + z.data) + a_0 * t.randn_like(z).data
    return z.detach()

def sample_langevin_posterior_notrick(z, x, G, E, K_1, a_1, loss_reconstruction ):
    z = z.clone().detach().requires_grad_(True)
    for i in range(K_1):
        x_hat = G(z)
        # g_log_lkhd = 1.0 / (2.0 * llhd_sigma * llhd_sigma) * mse(x_hat, x)
        g_log_lkhd = loss_reconstruction(x_hat, x).mean(dim=0)
        grad_g = t.autograd.grad(g_log_lkhd, z)[0]
        en = E(z)
        grad_e = t.autograd.grad(en.sum(), z)[0]
        z.data = z.data - 0.5 * a_1 * a_1 * (grad_g + grad_e + z.data) + a_1 * t.randn_like(z).data
    return z.detach()







