from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from torch import Tensor

from decoder import DecoderBase

class PopulationBase(nn.Module):
    name: str = ""

    def __init__(self, readout):
        super().__init__()
        self.readout = readout

    def forward(self, x: Tensor, mu: Tensor, sigma: Tensor, orientation: tuple[float, float]) -> tuple[Tensor, Tensor]:
        activation = self.activation(x, mu, sigma)
        return (activation, self.readout(activation, mu, orientation))

    def activation(self, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        pass

@dataclass
class CircularPopulationBase(nn.Module):
    name: str = ""

    def forward(self, x: Tensor, mu: Tensor, sigma: Tensor, orientation: tuple[float, float]) -> tuple[Tensor, Tensor]:
        activation = self.activation(x, mu, sigma, orientation)
        return (activation, self.readout(activation))

    def activation(self, x: Tensor, mu: Tensor, sigma: Tensor, orientation: tuple[float, float]) -> Tensor:
        pass

class Gaussian(PopulationBase):
    name = "gaussian"

    def activation(self, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (2 * sigma ** 2)

    
class TuningCurve(PopulationBase):
    name = "tuning_curve"

    def activation(self, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
class MexicanHat(PopulationBase):
    name = "mexican_hat"

    def activation(self, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return (1 - ((x - mu) / sigma) ** 2) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
class LogNormal(PopulationBase):
    name = "log_normal"

    def activation(self, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        x = torch.clamp(x, min=1e-10)

        coeff = 1.0 / (x * sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
        exponent = -((torch.log(x) - mu) ** 2) / (2 * sigma**2)

        return coeff * torch.exp(exponent)
    
class CircularTuningCurve(CircularPopulationBase):
    name = "circular_tuning_curve"

    def activation(self, x: Tensor, mu: Tensor, sigma: Tensor, orientation: tuple[float, float]) -> Tensor:
        L = orientation[1] - orientation[0]

        dist = torch.abs(x - mu)
        dist = torch.minimum(dist, L - dist)

        return torch.exp(-0.5 * (dist / sigma) ** 2)

class Distribution(Enum):
    ZERO_MEAN = 0,
    ZERO_BASE = 1,

class SineWave:
    def __call__(self, freq, phase, amp, x_pos, x_size, dist=Distribution.ZERO_MEAN):
        if dist == Distribution.ZERO_MEAN: 
            return self._sine(freq, phase, amp, x_pos, x_size)
        
        return self._sine_base(freq, phase, amp, x_pos, x_size)
        

    def _sine(self, freq, phase, amp, x_pos, x_size): 
        return amp * torch.sin(2 * torch.pi * freq * x_pos / x_size + phase)

    def _sine_base(self, freq, phase, amp, x_pos, x_size):
        return amp * (torch.sin(2 * torch.pi * freq * x_pos / x_size + phase) + 1)
        