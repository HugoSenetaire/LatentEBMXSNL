

import matplotlib.pyplot as plt
import torchvision as tv
import wandb

global_dic_error = {}

def log(step, dic_loss, logger, name=""):
  for key,value in dic_loss.items():
    logger.log({name+key:value},step=step)


def draw_samples(prior_0, langevin_prior, posterior, approximate_posterior, step, logger):

    if prior_0 is not None :
        fig = plt.figure(figsize=(10,10))
        grid_prior = tv.utils.make_grid(prior_0/2+0.5,)
        plt.imshow(grid_prior.detach().cpu().permute(1,2,0).numpy())
        plt.title("BaseDistribution")
        img= wandb.Image(fig, caption=f"Base Distribution {step}")
        logger.log({f"BaseDistribution.png": img},step=step)
    
    if langevin_prior is not None :
        fig = plt.figure(figsize=(10,10))
        grid_langevin_prior = tv.utils.make_grid(langevin_prior/2+0.5)
        plt.imshow(grid_langevin_prior.detach().cpu().permute(1,2,0).numpy())
        plt.title("Prior")
        img= wandb.Image(fig, caption=f"Prior {step}")
        logger.log({f"Prior.png": img},step=step)
    
    if posterior is not None :
        fig = plt.figure(figsize=(10,10))
        grid_langevin_posterior = tv.utils.make_grid(posterior/2+0.5)
        plt.imshow(grid_langevin_posterior.detach().cpu().permute(1,2,0).numpy())
        plt.title("Posterior")
        img= wandb.Image(fig, caption=f"Posterior {step}")
        logger.log({f"Posterior.png": img},step=step)

    if approximate_posterior is not None :
        fig = plt.figure(figsize=(10,10))
        grid_langevin_approximate_posterior = tv.utils.make_grid(approximate_posterior/2+0.5)
        plt.imshow(grid_langevin_approximate_posterior.detach().cpu().permute(1,2,0).numpy())
        plt.title("Approximate Posterior")
        img= wandb.Image(fig, caption=f"Approximate Posterior {step}")
        logger.log({f"Approximate Posterior.png": img},step=step)



