import torch
import torch.nn as nn
import torch.nn.functional as F

class PopulationCodedLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha=100.0, sigma=1.5, ):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.alpha = alpha
        self.sigma = sigma
        self.hidden_dim = hidden_dim

    def forward(self, x):
        a = F.relu(self.linear(x))
        print(f"   a: {a}") 

        p = F.softmax(self.alpha * a, dim=1)
        print(f"   p: {p}") 

        positions = torch.arange(self.hidden_dim, device=x.device).float().unsqueeze(0)
        mu = torch.sum(p * positions, dim=1, keepdim=True)

        print(f"  mu: {mu}")
        
        mask = torch.exp(-0.5 * ((positions - mu) / self.sigma) ** 2) / (2 * self.sigma ** 2)
        mask = mask / mask.max(dim=-1, keepdim=True).values
        print(f"mask: {mask}")

        masked_a = a * mask
        print(f" a_m: {masked_a}")

        return masked_a