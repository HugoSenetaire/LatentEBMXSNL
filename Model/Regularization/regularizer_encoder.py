import torch

def regularization_encoder(dic_param, encoder, cfg, logger, step):
        '''
        Compute different gradients and regularization terms given the energy or the loss.
        '''
        dic_loss = {}
        # Regularization
        if "mu" in dic_param.keys():
            if cfg.regularization_encoder.l2_mu is not None and cfg.regularization_encoder.l2_mu > 0:
                mu = dic_param["mu"]
                l2_mu = torch.norm(mu, p=2, dim=1).mean()
                dic_loss["loss_l2_mu"] = cfg.regularization_encoder.l2_mu * l2_mu
                logger.log({"penalty/l2_mu": l2_mu},step=step)
                logger.log({"penalty/regularization_l2_mu": dic_loss["loss_l2_mu"]},step=step)

        return dic_loss

