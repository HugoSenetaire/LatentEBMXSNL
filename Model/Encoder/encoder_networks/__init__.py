from .encoder_mnist import _Encoder_MNIST
from .encoder_svhn import _Encoder_SVHN


dic_gen = {
    "conv_mnist": _Encoder_MNIST,
    "conv_svhn": _Encoder_SVHN,
}

def get_encoder_network(network_name, ngf, nz, nc, multiplier=1):
    network_name = network_name
    new_nc = multiplier*nc # For instance if I want to model the sigma, I need to multiply the number of channels by two to get the mus as well.
    if network_name in dic_gen:
        return dic_gen[network_name](ngf, nz, new_nc,)
    else:
        raise ValueError("Generator not implemented")