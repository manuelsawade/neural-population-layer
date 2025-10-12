
from datetime import datetime
import random

import torch

from activations.neuron import PreferredStimulus
from datasets.mnist import MNIST
from decoder import CircularMeanDecoder, WeightedAverageDecoder
from populations import CircularTuningCurve, TuningCurve
from datasets.cifar10 import CIFAR10
from training import HyperParameter, LinearNetworkTraining, NeuronPopulationParameter, NeuronPopulationTraining


date_time = datetime.now()

datasets = [MNIST(), CIFAR10()]

runs = 20
epochs = 100

noise_classes = [0.0, 0.5]
hidden_dims = [128, 256]
batch_sizes=[50]
learning_rates=[0.00001]
weight_decays=[0.0001]

neurons_sizes=[4, 8, 12]
sigma_values=[0.4, 0.8, 1.2]
orientation_values=[(-1, 1), (-2.5, 2.5), (-4, 4)]
stimulus_values=[PreferredStimulus.LINEAR, PreferredStimulus.RAND_NORMAL]


activations=[ 
    TuningCurve(readout=WeightedAverageDecoder()), 
    CircularTuningCurve(readout=CircularMeanDecoder())]

all_runs = runs * len(noise_classes) * len(hidden_dims) * len(batch_sizes) * len(learning_rates) * len(weight_decays) * len(neurons_sizes) * len(sigma_values) * len(orientation_values) * len(stimulus_values) * len(activations)
all_run_count = 0

for dataset in datasets:
    for activation in activations:
        for batch_size in batch_sizes:
            for weight_decay in weight_decays:
                for learning_rate in learning_rates:
                    for training_noise in noise_classes:
                        for hidden_dim in hidden_dims:            
                            for neurons in neurons_sizes:
                                for sigma in sigma_values:
                                    for orientation in orientation_values:
                                        for stimulus in stimulus_values:                                                                                
                                            for i in range(runs):
                                                seed = random.randint(1000000, 9999999)
                                                torch.manual_seed(seed)  
                                                
                                                population_training = NeuronPopulationTraining(
                                                    hyper_parameter=NeuronPopulationParameter(
                                                        dataset=MNIST(),
                                                        hidden_dim=hidden_dim,
                                                        training_noise=training_noise,
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
                                                        activation=activation,
                                                        subset=None
                                                    )
                                                )

                                                population_training.run()

                                                print(f"Run {all_run_count}/{all_runs}")
                                                all_run_count = all_run_count+1
