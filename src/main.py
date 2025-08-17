import torch
import torch.nn as nn

from activations.masked import MaskedPopulation, Norm
from activations.neuron import NeuronPopulation, OutNorm, PreferredStimulus
from activations.sine_layer import SineLayerPopulationActivation
from data.lc25000 import LC25000, LC25000Dataset
from networks import NeuralNetwork
from populations import Distribution, Gaussian, LogNormal, MexicanHat, TuningCurve
from trainer import Trainer
from data.mnist import MNIST

training_noise = 1.0

input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

torch.manual_seed(100)

linear_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
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
    SineLayerPopulationActivation(
        freq=8.0, 
        amp=1.0,
        dist=Distribution.ZERO_BASE,),
    nn.Linear(hidden_dim, output_dim))

deep_fixed_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    SineLayerPopulationActivation(
        freq=16.0, 
        amp=1.0,
        grad_phase=True,
        grad_amp=True,
        dist=Distribution.ZERO_BASE,),
    nn.Linear(hidden_dim, hidden_dim),
    SineLayerPopulationActivation(
        freq=16.0, 
        amp=1.0,
        grad_phase=True,
        grad_amp=True,
        dist=Distribution.ZERO_BASE,),
    nn.Linear(hidden_dim, output_dim))

mexican_hat_stack = nn.Sequential(
    nn.Linear(input_dim, 200),
    NeuronPopulation(
        200, 
        sigma=0.15,
        neurons=6,
        orientation=(-2, 1),
        activation=MexicanHat(),
        stimulus=PreferredStimulus.LINEAR),
    nn.LazyLinear(output_dim),    
    )

hidden_dim = 200

pop_stack = nn.Sequential(
    NeuronPopulation(
        input_dim, 
        sigma=1.5,
        neurons=6,
        orientation=(0, 1),
        activation=TuningCurve(),
        stimulus=PreferredStimulus.COSINE),
    nn.LazyLinear(output_dim),    
    )

tuning_curve_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    NeuronPopulation(
        hidden_dim, 
        sigma=1.5,
        neurons=6,
        orientation=(0, 1),
        activation=TuningCurve(),
        stimulus=PreferredStimulus.COSINE),
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

for stack in [pop_stack]:    
    trainer = Trainer(
        model=NeuralNetwork(layers=stack),
        data_set=MNIST(),
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