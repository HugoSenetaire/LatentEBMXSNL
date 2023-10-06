import torch

def _get_adamw(cfg, net):
    optim = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.lr,
        betas=(cfg.b1, cfg.b2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    return optim

def _get_adam(cfg, net):
    optim = torch.optim.Adam(
        net.parameters(),
        lr=cfg.lr,
        betas=(cfg.b1, cfg.b2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    return optim

def get_optimizer(cfg_optim, net):
    if cfg_optim.optimizer_name == "adamw":
        return _get_adamw(cfg_optim, net)
    elif cfg_optim.optimizer_name == "adam":
        return _get_adam(cfg_optim, net)
    else:
        raise ValueError("Optimizer name not valid")
    
