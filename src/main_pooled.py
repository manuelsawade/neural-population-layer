
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

identifier = "mnist_cifar10_v1_0"

parameter = {
    "dataset": [MNIST(), CIFAR10()],
    "runs": [20],
    "epochs": [100],
    "noise": [0.0, 0.5],
    "hidden_dim": [200],
    "batch_size": [32],
    "neurons": [4, 8, 12],
    "sigma": [0.4, 0.8, 1.2],
    "orientation": [(-1, 1), (-2.5, 2.5), (-4, 4)],
    "stimulus": [PreferredStimulus.LINEAR, PreferredStimulus.RAND_NORMAL],
    "learning_rate": [0.00001],
    "weight_decay": [0.0001],
    "activation": [TuningCurve(readout=WeightedAverageDecoder())]#, CircularTuningCurve(readout=CircularMeanDecoder())],
}

all_parameter = list(itertools.product(*parameter.values()))
print(f"max runs: {len(all_parameter) * parameter["runs"][0]}")

def run(params: tuple):
    dataset, runs, epochs, noise, hidden_dim, batch_size, neurons, sigma, orientation, stimulus, learning_rate, weight_decay, activation = params
    for i in range(runs):
        seed = random.randint(1000000, 9999999)
        torch.manual_seed(seed)  
        
        population_training = NeuronPopulationTraining(
            hyper_parameter=NeuronPopulationParameter(
                dataset=dataset,
                hidden_dim=hidden_dim,
                training_noise=noise,
                test_noise=0.2,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs,
                created_on=date_time,
                sigma=sigma,
                orientation=orientation,
                stimulus=stimulus,
                neurons=neurons,
                seed=seed,
                subset=None,
                activation=activation,
                index=i,
                identifier=identifier))

        population_training.run()

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(run, all_parameter))
