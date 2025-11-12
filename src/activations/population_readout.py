from enum import StrEnum
import torch
import torch.nn as nn


class Variance(StrEnum):
    RAND_NORMAL = 'RAND_NORMAL',
    RAND_UNIFORM = 'RAND_UNIFORM',
    LINEAR = 'LINEAR',

class PopulationCodeActivation(nn.Module):
    
    def __init__(self, input_dim, neurons, deviation, variance, interval):
        super().__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.log_sigma = nn.Parameter(torch.log(torch.ones(input_dim, neurons) * deviation)) 
        self.interval = interval
        self.epsilon = 1e-8

        match variance:
            case Variance.RAND_NORMAL: 
                self.mu = nn.Parameter(torch.normal(mean=(interval[0] + interval[1]) / 2, std=1, size=(input_dim, neurons)))
            case Variance.RAND_UNIFORM: 
                self.mu = nn.Parameter(torch.tensor((interval[1] - interval[0]) * torch.rand(size=(input_dim, neurons)) + interval[0]))
            case Variance.LINEAR: 
                self.mu = nn.Parameter(torch.linspace(interval[0], interval[1], steps=neurons).unsqueeze(0).repeat(input_dim, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mu.unsqueeze(0) 
        sigma = torch.exp(self.log_sigma).unsqueeze(0)
        
        population_code = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

        denom = population_code.sum(dim=-1) + self.epsilon
        cont = (population_code * mu).sum(dim=-1, keepdim=False) / denom

        decoded_population = cont.clamp(self.interval[0], self.interval[1])

        return decoded_population

