import random
import torch
import torch.nn as nn

from activations.masked import MaskedPopulation, Norm
from activations.neuron import NeuronPopulation, OutNorm, PreferredStimulus
from activations.sine_layer import SineLayerPopulationActivation
from datasets.lc25000 import LC25000, LC25000Dataset
from networks import NeuralNetwork
from populations import Distribution, Gaussian, LogNormal, MexicanHat, TuningCurve
from trainer import Trainer
from datasets.mnist import MNIST

dataset = MNIST()
input_dim = dataset.input_dim
output_dim = dataset.output_dim

hidden_dim = 80
training_noise = 0.0

seed = 10000#random.randint(1000000, 9999999)
torch.manual_seed(seed)

tuning_curve_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    NeuronPopulation(
        hidden_dim, 
        sigma=1.5,
        neurons=10,
        orientation=(0, 2),
        activation=LogNormal(),
        stimulus=PreferredStimulus.LINEAR),
    nn.LazyLinear(output_dim),    
    )

trainer = Trainer(
    model=NeuralNetwork(layers=tuning_curve_stack),
    dataset=dataset,
    #subset=1000,
    training_noise=training_noise,
    batch_size=256,
    learning_rate=0.00001,
    weight_decay=0.0001
)

output = {}
output['training_noise'] = training_noise
output['seed'] = seed
output['stack'] = 'log_normal'
output['network'] = 'nn'

trainer.train(epochs=20)
trainer.test(noise=0.2, summary=output)

# TuningCurve, dim=50, neurons=8, sigma=0.3, orientation=(-0.5, 1.5), subset=1000
#   FSA_2.0: 0.446009 (mean)
#   FSA_2.0: 0.370049 (std)
#   FSD_2.0: 0.463652 (mean)
#   FSD_2.0: 0.052752 (std)
#   FSA_inf: 0.444557 (mean)
#   FSA_inf: 0.371403 (std)
#   FSD_inf: 0.858256 (mean)
#   FSD_inf: 0.022193 (std)

# dim=50, neurons=8, sigma=0.3, orientation=(-0.5, 0.5)
#   FSA_2.0: 0.368772 (std)
#   FSD_2.0: 0.062712 (std)
#   ROBY_2.0: 0.049814 (std)
#   FSA_inf: 0.373682 (std)
#   FSD_inf: 0.027637 (std)
#   ROBY_inf: 0.212016 (std)

#Ruby
#   FSA_2.0: 0.091087 (std)
#   FSD_2.0: 0.072321 (std)
#   ROBY_2.0: 0.068194 (std)
#   FSA_inf: 0.114317 (std)
#   FSD_inf: 0.090942 (std)
#   ROBY_inf: 0.090444 (std)