from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from populations import MexicanHat, TuningCurve

class OutNorm(Enum):
    NONE = 0,
    MEAN = 1,
    NORM = 2,
    LAYER_NORM = 3

class PreferredStimulus(Enum):
    RAND_NORMAL = 0,
    RAND_UNIFORM = 1,
    LINEAR = 2,
    COSINE = 3

class NeuronPopulation(nn.Module):
    def __init__(
            self, 
            input_dim, 
            sigma=0.5, 
            temp=0.5,
            out_norm=OutNorm.NONE,
            stimulus=PreferredStimulus.RAND_NORMAL,
            neurons = 10,
            orientation: tuple[float, float] = (-0.5, 0.5),
            activation = TuningCurve()):
        super().__init__()
        self.num_features = input_dim
        self.neurons_per_feature = neurons
        self.out_norm = out_norm
        self.log_sigma = nn.Parameter(torch.log(torch.ones(input_dim, neurons) * sigma)) 
        self.temp = temp
        self.ln = nn.LayerNorm(neurons)
        self.orientation = orientation
        self.activation = activation

        match stimulus:
            case PreferredStimulus.RAND_NORMAL: 
                self.mu = nn.Parameter(torch.normal(
                    mean=(orientation[0] + orientation[1]) / 2, 
                    std=1, 
                    size=(input_dim, neurons)))
            case PreferredStimulus.RAND_UNIFORM: 
                self.mu = nn.Parameter(torch.empty(input_dim, neurons))
                nn.init.uniform_(self.mu, orientation[0], orientation[1])
            case PreferredStimulus.LINEAR: 
                self.mu = nn.Parameter(torch.linspace(orientation[0], orientation[1], steps=neurons))
            case PreferredStimulus.COSINE:
                self.mu = nn.Parameter(self._cosine_spacing(neurons, orientation))
    
    def forward(self, x: torch.Tensor) -> None:
        x_expanded = x.unsqueeze(-1) 
        mu = self.mu.unsqueeze(0) 
        sigma = torch.exp(self.log_sigma).unsqueeze(0)
        
        # Kullback–Leibler divergence
        out = self.activation(x_expanded, mu, sigma)
        
        match self.out_norm:
            case OutNorm.MEAN: 
                out = out - out.mean(dim=1, keepdim=True)
            case OutNorm.NORM: 
                out = out / (out.sum(dim=-1, keepdim=True) + 1e-6)
            case OutNorm.LAYER_NORM: 
                out = self.ln(out)

        return out.view(x.size(0), self.num_features * self.neurons_per_feature)
    
    def _cosine_spacing(self, neurons: int, orientation: tuple[float, float]) -> torch.Tensor:
        theta = torch.linspace(0, torch.pi, steps=neurons)
        y = torch.cos(theta)

        y = orientation[0] + (y - y.min()) * (orientation[1] - orientation[0]) / (y.max() - y.min())
        return nn.Parameter(torch.tensor(y))