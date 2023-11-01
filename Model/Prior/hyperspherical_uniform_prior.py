import torch.nn as nn
import torch
from ..Utils.hyperspherical_utils import HypersphericalUniform


class HyperSphericalUniformPrior(nn.Module):
    def __init__(self, nz, cfg):
        super().__init__()
        self.cfg = cfg
        self.nz = nz
        self.aux_param = torch.nn.parameter.Parameter(torch.zeros((self.nz-1,)), requires_grad=False)

        
    def log_prob(self, z):
        z = z/z.norm(dim=-1, keepdim=True)
        self.distribution = HypersphericalUniform(self.nz-1, self.aux_param.device )
        return self.distribution.log_prob(z).reshape(z.shape[0],).to(self.aux_param.device)
    
    def sample(self, n):
        self.distribution = HypersphericalUniform(self.nz-1, self.aux_param.device )
        return self.distribution.sample(torch.Size((n,),)).to(self.aux_param.device)
    

    
    
