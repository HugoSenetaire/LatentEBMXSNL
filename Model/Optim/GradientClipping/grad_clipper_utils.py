import torch
import math

def clip_grad_adam(parameters, optimizer, nb_sigma = 3):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None or p.grad.data is None:
                    continue
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = nb_sigma * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))

def clip_grad_gebm(net,):
    for name, param in net.named_parameters():
        if 'log_partition' not in name:
            if param.grad is not None:
                new_grad = 2.*(param.grad.data)/(1+ (param.grad.data)**2)
                if math.isfinite(torch.norm(new_grad).item()):
                    param.grad.data = 1.*new_grad
                else:
                    print('nan grad')
                    param.grad.data = torch.zeros_like(new_grad)
        else :
            if param.grad is not None:
                new_grad = param.grad.data/(1-param.grad.data)
                if math.isfinite(torch.norm(new_grad).item()):
                    param.grad.data = new_grad
                else:
                    param.grad.data = torch.zeros_like(new_grad)
        

def grad_clipping(net, net_name, cfg, current_optim, logger, step):
        # Grad clipping
        clip_grad_type = cfg.clip_grad_type
        clip_grad_value = cfg.clip_grad_value
        nb_sigma = cfg.nb_sigma
        # replace_nan = cfg.replace_nan


        if clip_grad_type == "norm":
            if clip_grad_value is not None:
                logger.log({"train/{}_clip_grad_norm".format(net_name): clip_grad_value}, step=step)
                torch.nn.utils.clip_grad_norm_(
                    parameters=net.parameters(),
                    max_norm=clip_grad_value,
                )
        elif clip_grad_type == "abs":
            if clip_grad_value is not None:
                logger.log({"train/{}_clip_grad_abs".format(net_name): clip_grad_value}, step=step)
                torch.nn.utils.clip_grad_value_(
                    parameters=net.parameters(),
                    clip_value=clip_grad_value,
                )
        elif clip_grad_type == "adam":
            if nb_sigma is not None:
                logger.log({"train/{}_clip_grad_adam_nb_sigma".format(net_name): nb_sigma}, step=step)
                clip_grad_adam(net.parameters(),
                        current_optim,
                        nb_sigma=nb_sigma)
        elif clip_grad_type == "gebm":
            clip_grad_gebm(net, )
        elif clip_grad_type is None:
            pass
        else :
            raise NotImplementedError
        # if replace_nan:
            # for param in net.parameters():
                # if param.grad is not None and math.isfinite(torch.norm(param.grad).item()):
                    # param.grad.data = torch.zeros_like(param.grad.data)
