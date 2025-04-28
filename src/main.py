import torch.nn as nn

from layers import FixedPopulation
from layers.masked import MaskedPopulation
from layers.oscillatory import OscillatoryPopulation
from networks import NeuralNetwork
from trainer import Trainer
from data.data import MNIST

training_data, test_data = MNIST()(training_noise=0.3)

input_dim = 28 * 28
hidden_dim = 128
output_dim = 10

linear_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(), 
    nn.Linear(hidden_dim, output_dim))

masked_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    MaskedPopulation(),
    nn.ReLU(), 
    nn.Linear(hidden_dim, output_dim))

fixed_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    FixedPopulation(),
    nn.ReLU(), 
    nn.Linear(hidden_dim, output_dim))

oscillatory_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    OscillatoryPopulation(),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim))


for stack in [oscillatory_stack]: 
    model = NeuralNetwork(layers=stack)
    
    #print(f"Model: {stack}")
    trainer = Trainer(
        model=model,
        training_data=training_data, 
        test_data=test_data,
        batch_size=8
    )

    trainer.train(epochs=10)
    trainer.test()