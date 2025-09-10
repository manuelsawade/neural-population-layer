import torch
import torch.nn as nn
import random

from activations.neuron import NeuronPopulation, PreferredStimulus
from networks import NeuralNetwork
from populations import LogNormal
from trainer import Trainer
from datasets.mnist import MNIST

dataset = MNIST()
input_dim = dataset.input_dim
output_dim = dataset.output_dim

hidden_dim = 50
training_noise = 1.0
seed = random.randint(1000000, 9999999)
torch.manual_seed(seed)
   
trainer = Trainer(
    model=NeuralNetwork(layers=nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        NeuronPopulation(
            hidden_dim, 
            sigma=0.5,
            neurons=4,
            orientation=(0, 1),
            activation=LogNormal(),
            stimulus=PreferredStimulus.LINEAR),
        nn.LazyLinear(output_dim),    
    )),
    dataset=dataset,
    training_noise=training_noise,
    batch_size=32,
    learning_rate=0.0001,
)

output = {}
output['training_noise'] = training_noise
output['seed'] = seed
output['stack'] = 'log_normal'
output['network'] = 'nn'


trainer.train(epochs=20)
trainer.test(noise=0.2, summary=output)