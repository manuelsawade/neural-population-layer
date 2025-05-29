import torch
import torch.nn as nn

from layers.fixed import FixedPopulation
from layers.masked import Norm, MaskedPopulation
from networks import NeuralNetwork
from populations import Distribution
from trainer import Trainer
from data.mnist import MNIST

training_noise = 0.0

training_data, test_data = MNIST()(training_noise=training_noise)

input_dim = 28 * 28
hidden_dim = 128
output_dim = 10

#torch.manual_seed(100)

linear_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(), 
    nn.Linear(hidden_dim, output_dim))

masked_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    MaskedPopulation(
        freq=16.0, 
        amp=1, 
        norm=Norm.OUTPUT, 
        dist=Distribution.ZERO_BASE,
        scale_mask=False),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim))

fixed_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    FixedPopulation(
        freq=16.0, 
        phase=0.5,
        amp=1,
        grad_phase=False),
    nn.Linear(hidden_dim, output_dim))

for stack in [fixed_stack]:    
    trainer = Trainer(
        model=NeuralNetwork(layers=stack),
        training_data=training_data, 
        test_data=test_data,
        batch_size=32,
        learning_rate=0.0001,
    )

    output = {}
    output['Training Noise'] = training_noise
    output['Frequency'] = 16.0
    output['Amplitude'] = 4

    trainer.train(epochs=10)
    trainer.test(noise=0.2, summary=output)

# torch.Size([128])
# torch.Size([32, 128])
# torch.Size([32, 128])

# mask   torch.Size([128])
# diff   torch.Size([32, 128])
# num    torch.Size([128])
# denum  torch.Size([32])
# denum  torch.Size([32, 1])
# a      torch.Size([32, 128])
# a_norm torch.Size([32, 128])

# diff = x - mask
#         print(diff.shape)
#         numerator = torch.sum(diff ** 2, dim=-1, keepdim=True)
#         print(numerator.shape)
#         denom = torch.sum(x ** 2, dim=-1, keepdim=True) + torch.sum(mask ** 2, dim=-1, keepdim=True)
#         print(denom.shape)
#         # denom = denom.unsqueeze(1)
#         # print(denom.shape)

#         a = numerator / (denom + self.eps)
#         print(a.shape)
#         a_norm = a / a.max(dim=-1, keepdim=True).values
#         print(a_norm.shape)
#         return 0