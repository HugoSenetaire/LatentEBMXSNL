import torch

def clamp_norm(data, max_norm,):     
    if max_norm is not None:
        data_norm = data.flatten(1).norm(p=2, dim=1, keepdim=True)
        data = data / data_norm * torch.min(data_norm, torch.ones_like(data_norm) * max_norm)
    return data

def clamp_value(data, min_value=None, max_value=None):
    if min_value is not None:
        data = data.clamp(min=min_value)
    if max_value is not None:
        data = data.clamp(max=max_value)
    return data


def clamp_all(data, max_norm=None, min_value=None, max_value=None,):
    data = clamp_norm(data, max_norm,)
    data = clamp_value(data, min_value=min_value, max_value=max_value)
    return data