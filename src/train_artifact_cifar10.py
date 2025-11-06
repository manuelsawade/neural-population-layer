
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import itertools
import random

import torch

from activations.neuron import PreferredStimulus
from datasets.mnist import MNIST
from decoder import CircularMeanDecoder, WeightedAverageDecoder
from populations import CircularTuningCurve, TuningCurve
from datasets.cifar10 import CIFAR10
from training.training import NeuronPopulationParameter, NeuronPopulationTraining

date_time = datetime.now()

identifier = "cifar10_v2_0"

seed = random.randint(1000000, 9999999)
torch.manual_seed(seed)  

population_training = NeuronPopulationTraining(
    hyper_parameter=NeuronPopulationParameter(
        dataset=CIFAR10(),
        hidden_dim=512,
        training_noise=0.0,
        test_noise=0.2,
        batch_size=128,
        learning_rate=0.00032999967347189667,
        weight_decay=0.002778852099843325,
        epochs=100,
        created_on=date_time,
        sigma=0.6,
        orientation=(-4, 4),
        stimulus=PreferredStimulus.RAND_NORMAL,
        neurons=12,
        seed=seed,
        subset=None,
        activation=TuningCurve(readout=WeightedAverageDecoder()),
        index=0,
        identifier=identifier))

population_training.run()
