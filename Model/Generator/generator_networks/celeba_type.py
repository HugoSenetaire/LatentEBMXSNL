import torch.nn as nn

class ConvCELEBA(nn.Module):
    def __init__(self, ngf, nz, nc,):
        super().__init__()
        
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, ),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, ),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, ),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, ),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(ngf*2, ngf*1, 4, 2, 1, ),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(ngf*1, nc, 4, 2, 1),
        )

    def forward(self, z):
        return self.gen(z)
