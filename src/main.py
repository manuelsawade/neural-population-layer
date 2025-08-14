import torch
import torch.nn as nn

from data.lc25000 import LC25000, LC25000Dataset
from layers.fixed import FixedPopulation, GlobalNorm
from layers.masked import Norm, MaskedPopulation
from layers.population import PopulationEncoding, PopulationDecoding
from networks import NeuralNetwork
from populations import Distribution
from trainer import Trainer
from data.mnist import MNIST

training_noise = 0.0

input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

torch.manual_seed(100)

linear_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(), 
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(), 
    nn.Linear(hidden_dim, output_dim))

masked_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    MaskedPopulation(
        freq=16.0, 
        amp=1.0, 
        norm=Norm.NONE, 
        dist=Distribution.ZERO_MEAN,
        scale_mask=False),
    nn.ReLU(),  
    nn.Linear(hidden_dim, output_dim))

deep_masked_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    MaskedPopulation(
        freq=16.0, 
        amp=1.0, 
        norm=Norm.OUTPUT, 
        dist=Distribution.ZERO_MEAN,
        grad_phase=True,
        grad_amp=True,
        scale_mask=False),
    nn.ReLU(),  
    nn.Linear(hidden_dim, hidden_dim),
    MaskedPopulation(
        freq=16.0, 
        amp=1.0, 
        norm=Norm.OUTPUT, 
        dist=Distribution.ZERO_MEAN,
        grad_phase=True,
        grad_amp=True,
        scale_mask=False),
    nn.ReLU(),  
    nn.Linear(hidden_dim, output_dim))

fixed_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    FixedPopulation(
        freq=8.0, 
        amp=1.0,
        dist=Distribution.ZERO_BASE,),
    nn.Linear(hidden_dim, output_dim))

deep_fixed_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    FixedPopulation(
        freq=16.0, 
        amp=1.0,
        grad_phase=True,
        grad_amp=True,
        dist=Distribution.ZERO_BASE,),
    nn.Linear(hidden_dim, hidden_dim),
    FixedPopulation(
        freq=16.0, 
        amp=1.0,
        grad_phase=True,
        grad_amp=True,
        dist=Distribution.ZERO_BASE,),
    nn.Linear(hidden_dim, output_dim))

population_stack = nn.Sequential(
    PopulationEncoding(),
    nn.Linear(input_dim * 10, output_dim),
    PopulationDecoding())

for stack in [population_stack]:    
    trainer = Trainer(
        model=NeuralNetwork(layers=stack),
        data_set=MNIST(),
        training_noise=training_noise,
        batch_size=256,
        learning_rate=0.005,
    )

    output = {}
    output['Training Noise'] = training_noise
    # output['Frequency'] = 16.0
    # output['Amplitude'] = 4

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