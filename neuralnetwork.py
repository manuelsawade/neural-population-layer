import torch.nn as nn

from population_layer import PopulationCodedLayer

class PopulationNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.name = __class__.__name__
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            PopulationCodedLayer(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

class LinearNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.name = __class__.__name__
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits