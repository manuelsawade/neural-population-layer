import torch

from population_layer import PopulationCodedLayer

x = torch.tensor([[0.4, 0.3, 0.9, 0.3, 0.2]])
y = torch.randn(4, 5)
layer = PopulationCodedLayer(input_dim=5, hidden_dim=8)

output = layer(x)