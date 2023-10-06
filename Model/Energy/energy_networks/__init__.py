from .mnist_energy import E_MNIST
from .svhn_energy import E_SVHN

dic_energy = {
    "energy_mnist": E_MNIST,
    "energy_svhn": E_SVHN,
}

def get_energy_network(network_name, nz, ndf):
    network_name = network_name
    if network_name in dic_energy:
        return dic_energy[network_name](nz, ndf)
    else:
        raise ValueError("Energy not implemented")