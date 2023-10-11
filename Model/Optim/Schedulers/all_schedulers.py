
import torch

def get_exponential_lr(cfg_scheduler, opt):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg_scheduler.gamma)
    return scheduler


def get_scheduler(cfg_scheduler, opt):
    if cfg_scheduler.scheduler_name == "exponential_lr":
        return get_exponential_lr(cfg_scheduler, opt)
    else:
        raise ValueError("Optimizer name not valid")
    
