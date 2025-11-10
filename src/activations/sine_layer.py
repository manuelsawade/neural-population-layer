from enum import Enum
import torch
import torch.linalg as L
import torch.nn as nn
import torch.nn.functional as F

from populations import Distribution, MexicanHat, SineWave

class GlobalNorm(Enum):
    NONE = 0,
    MEAN = 1,
    SDT = 2

class SineLayerPopulationActivation(nn.Module):
    def __init__(self, freq=16.0, phase=0.0, amp=1.0, dist=Distribution.ZERO_MEAN, norm=GlobalNorm.NONE, grad_phase=False, grad_amp=False):
        super().__init__()
        self.freq = nn.Parameter(torch.tensor(freq))
        self.phase = nn.Parameter(torch.tensor(phase), requires_grad=grad_phase)
        self.amp = nn.Parameter(torch.tensor(amp), requires_grad=grad_amp)
        self.population = SineWave()
        self.dist = dist
        self.norm = norm
        self.eps = 1e-8

    def forward(self, x):
        x_size = x.size(dim=1)
        x_pos = torch.arange(x_size, device=x.device)

        preferred_values = self.population(self.freq, self.phase, self.amp, x_pos, x_size, self.dist)
        
        # diff = (x - preferred_values) ** 2
        # norm = torch.sum(x ** 2, dim=-1, keepdim=True) + torch.sum(preferred_values ** 2, dim=-1, keepdim=True)
        # norm = norm.mean(dim=-2, keepdim=True)
        # a = diff / (norm + self.eps)
        a_norm = a / a.max(dim=-1, keepdim=True).values
        
        return a_norm
    
class RandomPreferrenceActivation(nn.Module):
    def __init__(self, neurons, norm = GlobalNorm.MEAN):
        super().__init__()
        self.population = nn.Parameter(torch.rand((1, neurons)), requires_grad=False)
        self.norm: GlobalNorm = norm
        self.eps = 1e-8

    def forward(self, x):        
        diff = (x - self.population) ** 2
        norm = torch.sum(x ** 2, dim=-1, keepdim=True) + torch.sum(self.population ** 2, dim=-1, keepdim=True)

        match self.norm:
            case GlobalNorm.MEAN:
                norm = norm.mean(dim=-2, keepdim=True)
            case GlobalNorm.SDT:
                norm = norm.std(dim=-2, keepdim=True)

        a = diff / (norm + self.eps)
        a_norm = a / a.max(dim=-1, keepdim=True).values
        
        return a_norm