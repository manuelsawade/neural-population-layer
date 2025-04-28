import torch
import torch.linalg as L
import torch.nn as nn
import torch.nn.functional as F

from populations import MexicanHat

class FixedPopulation(nn.Module):
    def __init__(self, mu=63.5, sigma=32.0):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.population = MexicanHat()

    def forward(self, x):
        z_norm = x / x.max(dim=-1, keepdim=True).values

        positions = torch.arange(x.size(dim=1), device=x.device).float().unsqueeze(0)
        mask = self.population.mask(x=positions, mu=self.mu, sigma=self.sigma)
        mask = mask / mask.max(dim=-1, keepdim=True).values

        dists = ((x - mask) ** 2).sum(dim=-1)

        a = torch.exp(-0.5 * dists / (self.sigma ** 2)) 
        
        a = x + (mask - z_norm)
        #print(a)
        return a