from .mnist_type import ConvMnist
from .SVHN_type import ConvSVHN
from .celeba_type import ConvCELEBA


dic_gen = {
    "conv_mnist": ConvMnist,
    "conv_svhn": ConvSVHN,
    "conv_celeba": ConvCELEBA,
}

def get_generator_network(network_name, ngf, nz, nc, multiplier=1):
    network_name = network_name
    nc = multiplier*nc
    if network_name in dic_gen:
        return dic_gen[network_name](ngf, nz, nc,)
    else:
        raise ValueError("Generator not implemented")