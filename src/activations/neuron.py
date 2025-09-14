from enum import Enum
import torch
import torch.nn as nn
from torch import Tensor

from decoder import WeightedAverageDecoder
from populations import CircularPopulationBase, PopulationBase, TuningCurve

class PreferredStimulus(Enum):
    RAND_NORMAL = 0,
    RAND_UNIFORM = 1,
    LINEAR = 2,
    COSINE = 3

class NeuronPopulation(nn.Module):
    def __init__(
            self, 
            input_dim, 
            activation: PopulationBase | CircularPopulationBase = TuningCurve(readout=WeightedAverageDecoder()),
            sigma=0.5, 
            stimulus=PreferredStimulus.LINEAR,
            neurons = 10,
            orientation: tuple[float, float] = (-0.5, 0.5),
            ):
        super().__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.log_sigma = nn.Parameter(torch.log(torch.ones(input_dim, neurons) * sigma), requires_grad=True) 
        self.orientation = orientation
        self.activation = activation
        self.pop_out: torch.Tensor | None = None

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
                self.mu = nn.Parameter(torch.linspace(orientation[0], orientation[1], steps=neurons).unsqueeze(0).repeat(input_dim, 1))#.unsqueeze(0).repeat(input_dim, 1))
            case PreferredStimulus.COSINE:
                self.mu = nn.Parameter(self._cosine_spacing(neurons, orientation))

        self.mu.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1) 
        mu = self.mu.unsqueeze(0) 
        sigma = torch.exp(self.log_sigma).unsqueeze(0)
        
        encoded, decoded = self.activation(x_expanded, mu, sigma, self.orientation)
            
        self.pop_out = encoded.view(x.size(0), self.input_dim * self.neurons).detach()
        return decoded

# Todo: Check Kullback–Leibler divergence 
    
    def _cosine_spacing(self, neurons: int, orientation: tuple[float, float]) -> torch.Tensor:
        y = torch.cos(torch.linspace(0, torch.pi, steps=neurons))
        y = orientation[0] + (y - y.min()) * (orientation[1] - orientation[0]) / (y.max() - y.min())
        return nn.Parameter(torch.tensor(y))
