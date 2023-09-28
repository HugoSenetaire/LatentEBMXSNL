

import matplotlib.pyplot as plt
import torchvision as tv
import wandb

global_dic_error = {}

def log(step, dic_loss, logger, name=""):
  for key,value in dic_loss.items():
    logger.log({name+key:value},step=step)


transform_back_m1_1 = lambda x: (x)/2+0.5
transform_back_0_1 = lambda x: x

dic = {
   "m1_1": transform_back_m1_1,
    "0_1": transform_back_0_1,
}

def draw_samples(prior_0, langevin_prior, posterior, approximate_posterior, step, logger, transform_back_name="0_1", aux_name = ""):

    transform_back = dic[transform_back_name]
    if prior_0 is not None :
        fig = plt.figure(figsize=(10,10))
        grid_prior = tv.utils.make_grid(transform_back(prior_0),)
        plt.imshow(grid_prior.detach().cpu().permute(1,2,0).numpy())
        plt.title("BaseDistribution")
        img= wandb.Image(fig, caption=f"Base Distribution {step}")
        logger.log({aux_name+f"BaseDistribution.png": img},step=step)
        plt.close()

    
    if langevin_prior is not None :
        fig = plt.figure(figsize=(10,10))
        grid_langevin_prior = tv.utils.make_grid(transform_back(langevin_prior))
        plt.imshow(grid_langevin_prior.detach().cpu().permute(1,2,0).numpy())
        plt.title(aux_name+"Prior")
        img= wandb.Image(fig, caption=f"Prior {step}")
        logger.log({aux_name+f"Prior.png": img},step=step)
        plt.close()

    
    if posterior is not None :
        fig = plt.figure(figsize=(10,10))
        grid_langevin_posterior = tv.utils.make_grid(transform_back(posterior))
        plt.imshow(grid_langevin_posterior.detach().cpu().permute(1,2,0).numpy())
        plt.title(aux_name+"Posterior")
        img= wandb.Image(fig, caption=f"Posterior {step}")
        logger.log({aux_name+f"Posterior.png": img},step=step)
        plt.close()

    if approximate_posterior is not None :
        fig = plt.figure(figsize=(10,10))
        grid_langevin_approximate_posterior = tv.utils.make_grid(transform_back(approximate_posterior))
        plt.imshow(grid_langevin_approximate_posterior.detach().cpu().permute(1,2,0).numpy())
        plt.title(aux_name+"Approximate Posterior")
        img= wandb.Image(fig, caption=f"Approximate Posterior {step}")
        logger.log({aux_name+f"+ApproximatePosterior.png": img},step=step)
        plt.close()




