import torch.nn as nn
import torch 


class E_CELEBA(nn.Module):
  def __init__(self, nz, ndf):
        super().__init__()
        self.ebm = nn.Sequential(nn.Linear(nz, ndf), nn.LeakyReLU(0.2),
            nn.Linear(ndf, ndf), nn.LeakyReLU(0.2),
            nn.Linear(ndf, ndf), nn.LeakyReLU(0.2),
            nn.Linear(ndf, 1, bias=False))
        self.log_partition = nn.Parameter(torch.tensor(0.,),requires_grad=True)
  def forward(self, z):
      z_squeeze = z.squeeze()
      energy = self.ebm(z_squeeze)
      return energy+self.log_partition