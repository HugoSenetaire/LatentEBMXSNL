import math

import torch
import torch.nn as nn


class LossReconstruction():
    def __init__(self) -> None:
        self.multiplier_param = 1


    def __call__(self, x_hat, x):
        raise NotImplementedError


    def sample(self, param, return_mean = False):
        raise NotImplementedError

    

class FixedSigmaGaussianLossReconstruction(LossReconstruction):
    def __init__(self, sigma) -> None:
        super().__init__()
        self.sigma = sigma
        self.multiplier_param = 1


    def __call__(self, param, x):
        x_hat = param.flatten(1)
        x = x.flatten(1)
        return torch.sum((x_hat-x)**2, dim=1)/ (2.0 * self.sigma * self.sigma)  - math.log(self.sigma)* x.shape[1] - 0.5 * math.log(2*math.pi) * x.shape[1]

    def sample(self, param, return_mean = False):
        x_mu = param.flatten(1)
        x = torch.normal(x_mu, self.sigma)
        if return_mean:
            return x, x_mu
        else:
            return x

    

class LearnSigmaGaussianLossReconstruction(LossReconstruction):
    def __init__(self,) -> None:
        super().__init__()
        self.multiplier_param = 2

    def __call__(self, param, x):
        mu, log_sigma = param.flatten(1).chunk(2, dim=1)
        sigma = torch.exp(log_sigma)
        x = x.flatten(1)
        return torch.sum((mu-x)**2, dim=1)/ (2.0 * sigma * sigma)  - math.log(sigma)* x.shape[1] - 0.5 * math.log(2*math.pi) * x.shape[1]

    def sample(self, param, return_mean = False):
        x_mu, log_sigma = param.flatten(1).chunk(2, dim=1)
        sigma = torch.exp(log_sigma)
        x = torch.normal(x_mu, sigma)
        if return_mean:
            return x, x_mu
        else:
            return x

    
    

class BernoulliLossReconstruction(LossReconstruction):
    def __init__(self) -> None:
        super().__init__()
        self.multiplier_param = 1

    def __call__(self, param, x):
        p = param.flatten(1)
        x = x.flatten(1)
        if torch.isnan(p).any():
            raise ValueError
        return torch.sum(-x*torch.log(p+1e-8) - (1-x)*torch.log(1-p+1e-8), dim=1)

    def sample(self, param, return_mean = False):
        p = param.flatten(1)
        x = torch.bernoulli(p)
        if return_mean:
            return x, p
        else:
            return x
    

def get_loss_reconstruction(loss_name, llhd_sigma):
    if loss_name == "gaussian":
        return FixedSigmaGaussianLossReconstruction(llhd_sigma)
    elif loss_name == "gaussian_learn_sigma":
        return LearnSigmaGaussianLossReconstruction()
    elif loss_name == "bernoulli":
        return BernoulliLossReconstruction()
    else:
        raise NotImplementedError