import random
import torch
import torch.nn as nn

from activations.sine_layer import SineLayerPopulationActivation
from networks import NeuralNetwork
from trainer import Trainer
from datasets.mnist import MNIST

dataset = MNIST()
input_dim = dataset.input_dim
output_dim = dataset.output_dim

hidden_dim = 150
training_noise = 0.5

seed = random.randint(1000000, 9999999)
torch.manual_seed(seed)

trainer = Trainer(
    model=NeuralNetwork(layers=nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(), 
        nn.Linear(hidden_dim, output_dim))),
    dataset=dataset,
    training_noise=training_noise,
    batch_size=256,
    learning_rate=0.00001,
    weight_decay=0.0001
)

output = {}
output['training_noise'] = training_noise
output['stack'] = 'linear'
output['network'] = 'nn'
output['seed'] = seed

trainer.train(epochs=100)
trainer.test(noise=0.2, summary=output)