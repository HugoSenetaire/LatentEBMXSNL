import torch.nn as nn

class _Encoder_MNIST(nn.Module):
    def __init__(self, nef, nz, nc):
        super().__init__()
        
        self.conv_net = nn.Sequential(
                nn.Conv2d(in_channels=nc, out_channels=nef*2, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.3),
                nn.Conv2d(in_channels=nef*2, out_channels=nef*4, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.3),
                nn.Conv2d(in_channels=nef*4, out_channels=nef*8, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.3),
        )
        self.fc=nn.Sequential(
                nn.Linear(16*nef*8, nz),
                )


    def forward(self, x):
        x = self.conv_net(x)
        x = x.flatten(1)
        return self.fc(x)