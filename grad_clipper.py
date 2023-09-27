import torch

def clip_grad_adam(parameters, optimizer, nb_sigmas = 3):
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

                bound = nb_sigmas * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


def grad_clipping(net, net_name, cfg, current_optim, logger, step):
        # Grad clipping
        clip_grad_type = cfg[net_name+"_clip_grad_type"]
        clip_grad_value = cfg[net_name+"_clip_grad_value"]
        nb_sigmas = cfg[net_name+"_nb_sigma"]
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
            if nb_sigmas is not None:
                logger.log({"train/{}_clip_grad_adam_nb_sigmas".format(net_name): nb_sigmas}, step=step)
                clip_grad_adam(net.parameters(),
                        current_optim,
                        nb_sigmas=nb_sigmas)
        elif clip_grad_type is None:
            pass
        else :
            raise NotImplementedError

def grad_clipping_all_net(liste_network = [], liste_name = [], liste_optim = [], logger = None, cfg =None, step=None):
    for net, net_name, optim in zip(liste_network, liste_name, liste_optim):
      grad_clipping(net, net_name, cfg, optim, logger, step=step)



