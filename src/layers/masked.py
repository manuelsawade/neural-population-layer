import sys
import torch
import torch.nn as nn
from enum import Enum

from populations import Distribution, SineWave


class Norm(Enum):
    NONE = 0,
    INPUT = 1,
    OUTPUT = 2

class MaskedPopulation(nn.Module):
    def __init__(self, freq=16.0, phase=0.0, amp=1.0, norm=Norm.INPUT, dist=Distribution.ZERO_MEAN, grad_phase=False, grad_amp=False, scale_mask=False):
        super().__init__()
        self.freq = nn.Parameter(torch.tensor(freq), requires_grad=False)
        self.phase = nn.Parameter(torch.tensor(phase), requires_grad=grad_phase)
        self.amp = nn.Parameter(torch.tensor(amp), requires_grad=grad_amp)
        self.population = SineWave()
        self.norm = norm
        self.mask = dist
        self.scale_mask = scale_mask

    def forward(self, x):
        x_size = x.size(dim=1)
        x_pos = torch.arange(x_size, device=x.device)
        
        mask = self.population(self.freq, self.phase, self.amp, x_pos, x_size, self.mask)

        if self.scale_mask:
            mask = mask / x.max(dim=-1, keepdim=True).values
        
        match self.norm:
            case Norm.NONE:
                return self._forward(x, mask)
            case Norm.INPUT:
                return self._forward_norm_input(x, mask)
            case Norm.OUTPUT:
                return self._forward_norm_output(x, mask)       

    def _forward(self, x, mask):
        output = x + mask    
        return output
    
    def _forward_norm_input(self, x, mask):
        x_norm = x / x.max(dim=-1, keepdim=True).values
        output = x + (mask - x_norm)   
        return output
    
    def _forward_norm_output(self, x, mask):
        output = x + mask
        output = output / output.max(dim=-1, keepdim=True).values        
        return output