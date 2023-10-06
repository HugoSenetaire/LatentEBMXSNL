

import matplotlib.pyplot as plt
import torchvision as tv
import wandb


def log(step, dic_loss, logger, name=""):
  for key,value in dic_loss.items():
    logger.log({name+key:value},step=step)


transform_back_m1_1 = lambda x: (x.clamp(-1,1))/2+0.5
transform_back_0_1 = lambda x: x.clamp(0,1)

dic = {
   "m1_1": transform_back_m1_1,
    "0_1": transform_back_0_1,
}

def draw(img, step, logger, transform_back_name = "0_1", aux_name=""):
  transform_back = dic[transform_back_name]
  fig = plt.figure(figsize=(10,10))
  grid_prior = tv.utils.make_grid(transform_back(img),)
  plt.imshow(grid_prior.detach().cpu().permute(1,2,0).numpy())
  plt.title("BaseDistribution")
  img= wandb.Image(fig, caption=f"Base Distribution {step}")
  logger.log({aux_name+f"BaseDistribution.png": img},step=step)
  plt.close()

