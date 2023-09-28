
import torch
import torch.nn as nn


class _G_MNIST(nn.Module):
    def __init__(self, ngf, nz, nc):
        super().__init__()
        self.gen = nn.Sequential(nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0), nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf*8, ngf*4, 3, 2, 1), nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1), nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1), nn.Tanh())
    def forward(self, z):
        return self.gen(z)
    
class _G_BINARYMNIST(nn.Module):
    def __init__(self, ngf, nz, nc):
        super().__init__()
        self.gen = nn.Sequential(nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0), nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf*8, ngf*4, 3, 2, 1), nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1), nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1), )
    def forward(self, z):
        z = self.gen(z)
        x =torch.sigmoid(z).reshape(z.shape[0],-1,28,28)
        return x

class _G_SVHN(nn.Module):
    def __init__(self, ngf, nz, nc):
        super().__init__()
        self.gen = nn.Sequential(nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0), nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1), nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1), nn.LeakyReLU(),
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1), nn.Tanh())
    def forward(self, z):
        return self.gen(z)


class _Encoder_SVHN(nn.Module):
    def __init__(self, ngf, nz, nc):
        super().__init__()

        self.conv_net = nn.Sequential(
                nn.Conv2d(in_channels=nc, out_channels=ngf*2, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.3),
                nn.Conv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.3),
                nn.Conv2d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.3),
        )
        self.fc=nn.Sequential(
                # nn.Linear(16*ngf*8, 256),
                # nn.ReLU(),
                # nn.Linear(256, 2*nz),
                nn.Linear(16*ngf*8, 2*nz),
                )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.flatten(1)
        return self.fc(x)

class _E_MNIST(nn.Module):
    def __init__(self, nz, ndf):
        super().__init__()
        self.ebm = nn.Sequential(nn.Linear(nz, ndf), nn.LeakyReLU(0.2),
            nn.Linear(ndf, ndf), nn.LeakyReLU(0.2),
            # nn.Linear(ndf, ndf), nn.LeakyReLU(0.2),
            nn.Linear(ndf, 1, bias=False))
        self.log_partition = nn.Parameter(torch.tensor(0.,),requires_grad=True)
    def forward(self, z):
        return self.ebm(z.squeeze()).view(-1, 1, 1, 1)+self.log_partition

class _E_SVHN(nn.Module):
  def __init__(self, nz, ndf):
        super().__init__()
        self.mean = torch.nn.parameter.Parameter(torch.tensor(0.,),requires_grad=False)
        self.std = torch.nn.parameter.Parameter(torch.tensor(1.,),requires_grad=False)
        self.ebm = nn.Sequential(nn.Linear(nz, ndf), nn.LeakyReLU(0.2),
            nn.Linear(ndf, ndf), nn.LeakyReLU(0.2),
            nn.Linear(ndf, ndf), nn.LeakyReLU(0.2),
            nn.Linear(ndf, 1, bias=False))
        self.log_partition = nn.Parameter(torch.tensor(0.,),requires_grad=True)
  def forward(self, z):
      z_squeeze = z.squeeze()
      energy = self.ebm(z_squeeze)
      base_dist = torch.distributions.normal.Normal(self.mean, self.std).log_prob(z_squeeze).detach()
      base_dist = base_dist.reshape(z.shape).flatten(1).sum(1,).reshape(-1,1)
      return (energy-base_dist).view(-1, 1, 1, 1)+self.log_partition


class _Encoder_MNIST(nn.Module):
    def __init__(self, ngf, nz, nc):
        super().__init__()
        self.conv_net = nn.Sequential(
                nn.Conv2d(in_channels=nc, out_channels=ngf*2, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.3),
                nn.Conv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.3),
                nn.Conv2d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.3),
        )
        self.fc=nn.Sequential(
                # nn.Linear(16*ngf*8, 256),
                # nn.ReLU(),
                # nn.Linear(256, 2*nz),
                nn.Linear(16*ngf*8, 2*nz),
                )


    def forward(self, x):
        x = self.conv_net(x)
        x = x.flatten(1)
        return self.fc(x)

def network_getter(dataset, cfg):
    if dataset == "MNIST":
        _G = _G_MNIST(cfg['ngf'], cfg['nz'], cfg['nc'])
        _Encoder = _Encoder_MNIST(cfg['ngf'], cfg['nz'], cfg['nc'])
        _E = _E_MNIST(cfg['nz'], cfg['ndf'])
    elif dataset.startswith("SVHN"):
        _G = _G_SVHN(cfg['ngf'], cfg['nz'], cfg['nc'])
        _Encoder = _Encoder_SVHN(cfg['ngf'], cfg['nz'], cfg['nc'])
        _E = _E_SVHN(cfg['nz'], cfg['ndf'])
    elif dataset =="BINARYMNIST" :
        _G = _G_BINARYMNIST(cfg['ngf'], cfg['nz'], cfg['nc'])
        _Encoder = _Encoder_MNIST(cfg['ngf'], cfg['nz'], cfg['nc'])
        _E = _E_MNIST(cfg['nz'], cfg['ndf'])
    else :
        raise NotImplementedError()
    
    return _G, _Encoder, _E

