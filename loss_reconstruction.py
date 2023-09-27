import torch


class LossReconstruction():
    def __init__(self) -> None:
        pass

    def __call__(self, x_hat, x):
        raise NotImplementedError
    

class GaussianLossReconstruction(LossReconstruction):
    def __init__(self, sigma) -> None:
        super().__init__()
        self.sigma = sigma

    def __call__(self, x_hat, x):
        x_hat = x_hat.flatten(1)
        x = x.flatten(1)
        return torch.sum((x_hat-x)**2, dim=1)/ (2.0 * self.sigma * self.sigma) 

class BernoulliLossReconstruction(LossReconstruction):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x_hat, x):
        x_hat = x_hat.flatten(1)
        x = x.flatten(1)
        return torch.sum(-x*torch.log(x_hat+1e-8) - (1-x)*torch.log(1-x_hat+1e-8), dim=1)
    

def get_loss_reconstruction(cfg):
    loss_name= cfg["loss_reconstruction"]
    if loss_name == "gaussian":
        return GaussianLossReconstruction(cfg["llhd_sigma"])
    elif loss_name == "bernoulli":
        return BernoulliLossReconstruction()
    else:
        raise NotImplementedError