import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from populations import Oscillatory

class OscillatoryPopulation(nn.Module):
    def __init__(self, freq=16.0, phase=0.0):
        super().__init__()
        self.freq = nn.Parameter(torch.tensor(freq), requires_grad=False)
        self.phase = nn.Parameter(torch.tensor(phase), requires_grad=True)

    def forward(self, x):
        x_size = x.size(dim=1)
        x_norm = x / x.max(dim=-1, keepdim=True).values

        positions = torch.arange(x_size, device=x.device).float()

        theta = 2 * torch.pi * self.freq * positions / x_size + self.phase
        sine_mask = (torch.sin(theta) + 1) / 2

        sine_mask = sine_mask / sine_mask.max(dim=-1, keepdim=True).values

        sys.stdout.write(f'\r{self.freq.item():.4f}, {self.phase.item():.4f}')
        sys.stdout.flush()

        return x + (sine_mask - x_norm)