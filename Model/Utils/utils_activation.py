import torch.nn as nn

def get_activation(activation_name):
    if activation_name == None :
        return None
    elif activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "tanh":
        return nn.Tanh()
    else :
        raise NotImplementedError("Activation function {} not implemented".format(activation_name))