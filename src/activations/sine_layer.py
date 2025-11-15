from enum import StrEnum
import torch
import torch.nn as nn

from populations import SineWave


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

        activation = 1 - (distance / distance.max(dim=-1, keepdim=True).values)  
        return activation