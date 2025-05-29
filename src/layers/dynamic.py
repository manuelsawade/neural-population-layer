import torch
import torch.linalg as L
import torch.nn as nn
import torch.nn.functional as F

from populations import MexicanHat

class DynamicPopulation(nn.Module):
    def __init__(self, alpha=100.0, sigma=32.0):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.population = MexicanHat()

    def forward(self, x):
        p = F.softmax(self.alpha * x, dim=1)

        positions = torch.arange(x.size(dim=1), device=x.device).float().unsqueeze(0)
        mu = torch.sum(p * positions, dim=1, keepdim=True)
       
        mask = self.population.mask(x=positions, mu=mu, sigma=self.sigma)
        mask = mask / mask.max(dim=-1, keepdim=True).values
        
        a_norm = x / x.max(dim=-1, keepdim=True).values       
        masked_a = x + (mask - a_norm)
        
        return masked_a