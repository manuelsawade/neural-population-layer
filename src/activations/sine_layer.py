from enum import Enum, StrEnum
import torch
import torch.linalg as L
import torch.nn as nn
import torch.nn.functional as F

from populations import Distribution, MexicanHat, SineWave

class GlobalNorm(Enum):
    NONE = 0,
    MEAN = 1,
    SDT = 2

class PreferredValueInitializer(StrEnum):
    SINE_WAVE = "Sine Wave"
    RANDOM_NORMAL = "Random Normal"
    RANDOM_UNIFORM = "Random Uniform"

class PreferredValueActivation(nn.Module):
    def __init__(self, initialized: SineWave | torch.Tensor):
        super().__init__()
        self.preferred_values = initialized
        self.computed_preference = True if isinstance(initialized, SineWave) else False
        self.eps = 1e-8

    def forward(self, x):
        preferred_values = self.preferred_values() if self.computed_preference else self.preferred_values

        diff = (x - preferred_values) ** 2

        feature_norm = torch.sum(x ** 2, dim=-1, keepdim=True) + torch.sum(preferred_values ** 2, dim=-1, keepdim=True)
        layer_norm = feature_norm.mean(dim=-2, keepdim=True)

        distance = diff / (layer_norm + self.eps)

        activation = distance / distance.max(dim=-1, keepdim=True).values       
        return activation
    
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