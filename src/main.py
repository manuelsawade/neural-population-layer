
from datetime import datetime
import random

import torch

from activations.neuron import PreferredStimulus
from datasets.mnist import MNIST
from decoder import WeightedAverageDecoder
from populations import TuningCurve
from training import HyperParameter, LinearNetworkTraining, NeuronPopulationParameter, NeuronPopulationTraining


date_time = datetime.now()

noise_classes = [0.0, 0.3]
hidden_dims = [200]
subset = 10000
batch_size=32

runs = 5
epochs = 100

for training_noise in noise_classes:
    for hidden_dim in hidden_dims:                    
        for i in range(runs):
            seed = random.randint(1000000, 9999999)
            torch.manual_seed(seed)  
            
            linear_training = LinearNetworkTraining(
                hyper_parameter=HyperParameter(
                    dataset=MNIST(),
                    hidden_dim=hidden_dim,
                    training_noise=training_noise,
                    test_noise=0.2,
                    batch_size=batch_size,
                    learning_rate=0.00001,
                    weight_decay=0.0001,
                    epochs=epochs,
                    seed=seed,
                    created_on=date_time,
                    subset=subset
                )
            )

            linear_training.run()

for training_noise in noise_classes:
    for hidden_dim in hidden_dims:            
        for neurons in [4, 8, 12]:
            for sigma in [0.4, 0.8, 1.2]:
                for orientation in [(-1, 1), (-2, 2), (-3, 3), (-4, 4)]:
                    for stimulus in [PreferredStimulus.LINEAR, PreferredStimulus.RAND_UNIFORM, PreferredStimulus.RAND_NORMAL]:                                                                                
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
                                learning_rate=0.00002,
                                weight_decay=0.0001,
                                epochs=epochs,
                                created_on=date_time,
                                sigma=sigma,
                                orientation=orientation,
                                stimulus=stimulus,
                                neurons=neurons,
                                seed=seed,
                                subset=subset,
                                activation=TuningCurve(readout=WeightedAverageDecoder())
                                )
                            )

                            population_training.run()
