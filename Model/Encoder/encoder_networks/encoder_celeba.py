import torch.nn as nn

class _Encoder_CELEBA(nn.Module):
    def __init__(self, nef, nz, nc):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, 1),
            nn.LeakyReLU(),

            nn.Conv2d(nef*1, nef*2, 4, 2, 1, ),
            nn.LeakyReLU(),


            nn.Conv2d(nef*2, nef*4, 4, 2, 1, ),
            nn.LeakyReLU(),


            nn.Conv2d(nef*4, nef*8, 4, 2, 1, ),
            nn.LeakyReLU(),


            nn.Conv2d(nef*8, nef*16, 4, 2, 1, ),
            nn.LeakyReLU(),

            nn.Conv2d(nef*16, nz, 4, 1, 0, ),
        )
        self.fc=nn.Sequential(
                nn.Linear(nz, nz),
                )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.flatten(1)
        return self.fc(x)