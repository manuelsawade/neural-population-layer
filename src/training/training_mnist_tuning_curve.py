import random
import torch
import torch.nn as nn

from activations.masked import MaskedPopulation, Norm
from activations.neuron import NeuronPopulation, PreferredStimulus
from activations.sine_layer import SineLayerPopulationActivation
from datasets.lc25000 import LC25000, LC25000Dataset
from networks import NeuralNetwork
from populations import Distribution, Gaussian, LogNormal, MexicanHat, TuningCurve
from decoder import WeightedAverageDecoder
from trainer import Trainer
from datasets.mnist import MNIST

dataset = MNIST()
input_dim = dataset.input_dim
output_dim = dataset.output_dim

hidden_dim = 200
training_noise = 0.5

for i in range(1):
    #seed = random.randint(1000000, 9999999)
    #torch.manual_seed(seed)

    tuning_curve_stack = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        NeuronPopulation(
            hidden_dim, 
            sigma=0.15,
            neurons=8,
            orientation=(-4, 4),
            activation=TuningCurve(readout=WeightedAverageDecoder()),
            stimulus=PreferredStimulus.LINEAR),
        nn.LazyLinear(output_dim),    
        )
   
    trainer = Trainer(
        model=NeuralNetwork(layers=tuning_curve_stack),
        dataset=dataset,
        training_noise=training_noise,
        batch_size=256,
        learning_rate=0.00001,
        weight_decay=0.001
    )

    output = {}
    output['training_noise'] = training_noise
    #output['seed'] = seed
    output['stack'] = 'tuning_curve'
    output['network'] = 'nn'

    trainer.train(epochs=100)
    trainer.test(noise=0.2, summary=output)