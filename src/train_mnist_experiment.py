import torch
import torch.nn as nn

from activations.neuron import BayesianDecoder, NeuronPopulation, PreferredStimulus, WeightedAverageDecoder
from networks import NeuralNetwork
from populations import TuningCurve
from trainer import Trainer
from datasets.mnist import MNIST

dataset = MNIST()
input_dim = dataset.input_dim
output_dim = dataset.output_dim

hidden_dim = 150
training_noise = 0.5

seed = 10000#random.randint(1000000, 9999999)
torch.manual_seed(seed)

tuning_curve_stack = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    NeuronPopulation(
        hidden_dim, 
        sigma=1.5,
        neurons=10,
        orientation=(-4, 4),
        activation=TuningCurve(),
        decoder=WeightedAverageDecoder(),
        stimulus=PreferredStimulus.RAND_UNIFORM),
    nn.LazyLinear(10),     
    )

trainer = Trainer(
    model=NeuralNetwork(layers=tuning_curve_stack),
    dataset=dataset,
    training_noise=training_noise,
    batch_size=256,
    learning_rate=0.00001,
    weight_decay=0.0001
)

output = {}
output['training_noise'] = training_noise
output['seed'] = seed
output['stack'] = 'tuning_curve'
output['network'] = 'nn'

trainer.train(epochs=100)
trainer.test(noise=0.2, summary=output, write_file=True)

# linear
# Activations
#   layers.0: 10.491575 (mean), 1.518165 (std)
#   layers.1: 15.607875 (mean), 1.132015 (std)
#   layers.2: -11.978636 (mean), 4.487495 (std)
# Sharpness
#   layers.0.weight: 0.291127
#   layers.0.bias: 0.002701
#   layers.2.weight: 0.865868
#   layers.2.bias: 0.000698
# Roby
#   fsa_inf: 0.385335 (mean)
#   fsa_inf: 0.091983 (std)
#   fsd_inf: 0.573541 (mean)
#   fsd_inf: 0.093961 (std)
# Noise Sensitivity
#   fgsm: 0.070312 (mean)
#   fgsm: 0.046559 (std)

# TuningCurve, dim=80, neurons=10, sigma=1.5, orientation=(-4, 4),
#   FSA_2.0: 0.399347 (mean)
#   FSA_2.0: 0.077968 (std)
#   FSA_inf: 0.431301 (mean)
#   FSA_inf: 0.103777 (std)
#   FSD_2.0: 0.520517 (mean)
#   FSD_2.0: 0.054428 (std)
#   FSD_inf: 0.604029 (mean)
#   FSD_inf: 0.079470 (std)

# TuningCurve, dim=50, neurons=15, sigma=0.6, orientation=(-2.2, 2.2),
#   FSA_2.0: 0.314861 (mean)
#   FSA_2.0: 0.058440 (std)
#   FSA_inf: 0.399695 (mean)
#   FSA_inf: 0.094012 (std)
#   FSD_2.0: 0.479779 (mean)
#   FSD_2.0: 0.039953 (std)
#   FSD_inf: 0.541769 (mean)
#   FSD_inf: 0.098203 (std)

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

# Start
# FSA_2.0: 0.356583 (mean) 
# FSA_2.0: 0.087752 (std) 
# FSA_inf: 0.339027 (mean) 
# FSA_inf: 0.110048 (std) 
# FSD_2.0: 0.503080 (mean)
# FSD_2.0: 0.073084 (std) 
# FSD_inf: 0.865966 (mean) 
# FSD_inf: 0.073231 (std) 
