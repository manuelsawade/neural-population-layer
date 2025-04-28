import torch
import torch.nn as nn
import torch.nn.functional as F

from populations import Gaussian, MexicanHat

class PopulationCodedLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha=100.0, sigma=32.0, debug=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.alpha = alpha
        self.sigma = sigma
        self.hidden_dim = hidden_dim
        self.population = MexicanHat()
        self.debug = debug

    def forward(self, x):
        a =  F.relu(self.linear(x))
        
        print(f"   a: {a}") if self.debug else None

        p = F.softmax(self.alpha * a, dim=1)
        
        print(f"   p: {p}") if self.debug else None

        positions = torch.arange(self.hidden_dim, device=x.device).float().unsqueeze(0)
        mu = torch.sum(p * positions, dim=1, keepdim=True)

        print(f"  mu: {mu}") if self.debug else None
        
        mask = self.population.mask(x=positions, mu=mu, sigma=self.sigma)
        mask = mask / mask.max(dim=-1, keepdim=True).values
        
        print(f"mask: {mask}") if self.debug else None

        a_norm = a / a.max(dim=-1, keepdim=True).values
        
        print(f" a_n: {a_norm}") if self.debug else None

        masked_a = a + (mask - a_norm)
        
        print(f" a_m: {masked_a}") if self.debug else None

        return masked_a