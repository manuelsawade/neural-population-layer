import torch
import torch.nn as nn

from activations.masked import MaskedPopulation, Norm
from activations.neuron import NeuronPopulation, OutNorm, PreferredStimulus
from activations.sine_layer import SineLayerPopulationActivation
from datasets.lc25000 import LC25000, LC25000Dataset
from networks import NeuralNetwork
from populations import Distribution, Gaussian, LogNormal, MexicanHat, TuningCurve
from trainer import Trainer
from datasets.cifar10 import CIFAR10

dataset = CIFAR10()
input_dim = dataset.input_dim
output_dim = dataset.output_dim

hidden_dim = 200
training_noise = 1.0

torch.manual_seed(100)

linear_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(), 
    nn.Linear(hidden_dim, output_dim))

tuning_curve_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    NeuronPopulation(
        hidden_dim, 
        sigma=0.15,
        neurons=8,
        orientation=(0, 1),
        activation=TuningCurve(),
        stimulus=PreferredStimulus.LINEAR),
    nn.LazyLinear(output_dim),    
    )

log_normal_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    NeuronPopulation(
        hidden_dim, 
        sigma=0.75,
        neurons=8,
        orientation=(0, 1),
        activation=LogNormal(),
        stimulus=PreferredStimulus.LINEAR),
    nn.LazyLinear(output_dim),    
    )

for stack in [tuning_curve_stack]:    
    trainer = Trainer(
        model=NeuralNetwork(layers=stack),
        dataset=dataset,
        training_noise=training_noise,
        batch_size=32,
        learning_rate=0.0001,
    )

    output = {}
    output['Training Noise'] = training_noise
    # output['Frequency'] = 16.0
    # output['Amplitude'] = 4

    trainer.train(epochs=20)
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