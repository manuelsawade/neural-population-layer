
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

identifier = "cifar10_v3_0"

for i in range(1):
    for noise in [1.0]:
        seed = random.randint(1000000, 9999999)
        torch.manual_seed(seed)  

        population_training = NeuronPopulationTraining(
            hyper_parameter=NeuronPopulationParameter(
                dataset=CIFAR10(),
                hidden_dim=512,
                training_noise=noise,
                test_noise=0.2,
                batch_size=128,
                learning_rate=0.00035788469322176747,
                weight_decay=3.668390331265183e-06,
                epochs=100,
                created_on=date_time,
                sigma=1.0504721563645925,
                orientation=(-2, 2),
                stimulus=PreferredStimulus.RAND_NORMAL,
                neurons=14,
                seed=seed,
                subset=None,
                activation=TuningCurve(readout=WeightedAverageDecoder()),
                index=i,
                identifier=identifier))

        population_training.run()
