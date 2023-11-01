from .encoder_mnist import _Encoder_MNIST
from .encoder_svhn import _Encoder_SVHN
from .encoder_celeba import _Encoder_CELEBA


dic_gen = {
    "conv_mnist": _Encoder_MNIST,
    "conv_svhn": _Encoder_SVHN,
    "conv_celeba": _Encoder_CELEBA,
}

def get_encoder_network(network_name, ngf, nz, nc, lambda_nz=lambda x:x):
    network_name = network_name
    new_nz = lambda_nz(nz)# For instance if I want to model the sigma, I need to multiply the number of channels by two to get the mus as well.
    if network_name in dic_gen:
        return dic_gen[network_name](ngf, new_nz, nc,)
    else:
        raise ValueError("Generator not implemented")