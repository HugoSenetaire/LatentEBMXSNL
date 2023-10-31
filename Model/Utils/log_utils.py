

import matplotlib.pyplot as plt
import torchvision as tv
import wandb
import numpy as np 


def log(step, dic_loss, logger, name=""):
  for key,value in dic_loss.items():
    try :
      logger.log({name+key:value.mean().item()},step=step)
    except :
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
  grid_prior = tv.utils.make_grid(transform_back(img), normalize=True, nrow=int(np.sqrt(img.shape[0])))
  plt.imshow(grid_prior.detach().cpu().permute(1,2,0).numpy())
  plt.title(f"{aux_name}")
  img= wandb.Image(fig, caption=f"{aux_name}")
  logger.log({f"{aux_name}.png": img},step=step)
  plt.close()

def get_extremum(liste = []):
  if len(liste) == 0:
    raise ValueError("Empty list")
  else:
    current_min_x = 0
    current_max_x = 0
    current_min_y = 0
    current_max_y = 0
    for tensor in liste :
      current_min_x = min(current_min_x, tensor[:,0].min().item())
      current_max_x = max(current_max_x, tensor[:,0].max().item())
      current_min_y = min(current_min_y, tensor[:,1].min().item())
      current_max_y = max(current_max_y, tensor[:,1].max().item())

    return current_min_x, current_max_x, current_min_y, current_max_y




def plot_contour(sample, energy_list, energy_list_names, x, y, title, logger, step):
  if sample is not None :
    sample = sample.detach().cpu().numpy()
  fig, axs = plt.subplots(nrows=1, ncols= len(energy_list), figsize=(len(energy_list)*10, 10))
  for k,energy in enumerate(energy_list):
    energy = energy.detach().cpu().numpy()
    fig.colorbar(axs[k].contourf(x,y, energy,), ax=axs[k])
    if sample is not None :
      axs[k].scatter(sample[:,0], sample[:,1], c="red", s=1, alpha=0.5)
    axs[k].set_title(energy_list_names[k])
  fig.suptitle(title)
  img = wandb.Image(fig, caption=title)
  logger.log({f"{title}.png": img}, step=step)
  plt.close(fig=fig)