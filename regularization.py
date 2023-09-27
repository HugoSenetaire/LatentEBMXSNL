import torch
from torch.autograd import grad as torch_grad


def wgan_gradient_penalty(ebm, x, x_gen,):
    batch_size = x.size()[0]
    min_data_len = min(batch_size,x_gen.size()[0])
    # Calculate interpolation
    epsilon = torch.rand(min_data_len, device=x.device)
    for _ in range(len(x.shape) - 1):
        epsilon = epsilon.unsqueeze(-1)
    epsilon = epsilon.expand(min_data_len, *x.shape[1:])
    epsilon = epsilon.to(x.device)
    interpolated = epsilon*x.data[:min_data_len] + (1-epsilon)*x_gen.data[:min_data_len]
    interpolated = interpolated.detach()
    interpolated.requires_grad_(True)

    # Calculate probability of interpolated examples
    prob_interpolated = ebm.f_theta(interpolated).flatten(1).sum(1)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(x.device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(min_data_len, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sum(gradients ** 2, dim=1).mean()
    return gradients_norm

def regularization(ebm, x, x_gen, energy_data, energy_samples, cfg, logger, step):
        '''
        Compute different gradients and regularization terms given the energy or the loss.
        '''
        dic_loss = {}
        # Regularization
        if cfg["l2_grad"] is not None and cfg["l2_grad"] > 0:
            grad_norm = wgan_gradient_penalty(ebm, x, x_gen)
            dic_loss["l2_grad"] = cfg["l2_grad"] * grad_norm
            logger.log({"penalty/grad_norm": grad_norm},step=step)
            logger.log({"penalty/regularization_l2_grad": dic_loss["l2_grad"]},step=step)

        if cfg["l2_output"] is not None and cfg["l2_output"] > 0:
            l2_output = ((energy_data**2).mean() + (energy_samples**2).mean())
            dic_loss["loss_l2_output"] = cfg["l2_output"] * l2_output
            logger.log({"penalty/l2_output": l2_output},step=step)
            logger.log({"penalty/regularization_l2_output": dic_loss["loss_l2_output"]},step=step)

        if cfg["l2_param"] is not None and cfg["l2_param"] > 0:
            penalty = 0.
            len_params = 0.
            for params in ebm.parameters():
                len_params += params.numel()
                penalty += torch.sum(params**2)
            penalty = penalty / len_params
            dic_loss["loss_l2_param"] = cfg["l2_param"] * penalty
            logger.log({"penalty/l2_param": penalty},step=step)
            logger.log({"penalty/regularization_l2_param": dic_loss["loss_l2_param"]},step=step)

        return dic_loss

