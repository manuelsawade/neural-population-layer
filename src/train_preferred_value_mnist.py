
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import random

import torch

from activations.neuron import PreferredStimulus
from datasets.mnist import MNIST
from decoder import CircularMeanDecoder, WeightedAverageDecoder
from populations import CircularTuningCurve, Distribution, TuningCurve
from datasets.cifar10 import CIFAR10
from training.training import NeuronPopulationParameter, NeuronPopulationTraining, PreferredValueParameter, PreferredValueTraining

date_time = datetime.now()

identifier = "concept_preferred_value_v1_0"

try:
    #with open("./experiments/tuning/population_last-5-avg_cifar10_evaluation_2025_11_10_20_37_06.json", "r") as f:
        #tuning_result = json.loads(f.read())
        tuning_result = {
            "hidden_dim": 256,
            "noise": 0.0,
            "batch_size": 128,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "freq": 8.0,
            "phase": 0.0,
            "amp": 1.0,
            "dist": Distribution.ZERO_MEAN,
            "requires_grad": False,
        }

        for i in range(1):
            seed = random.randint(1000000, 9999999)
            torch.manual_seed(seed)  

            population_training = PreferredValueTraining(
                hyper_parameter=PreferredValueParameter(
                    stack="population",
                    dataset=MNIST(),
                    hidden_dim=tuning_result["hidden_dim"],
                    training_noise=tuning_result["noise"],
                    test_noise=0.2,
                    batch_size=tuning_result["batch_size"],
                    learning_rate=tuning_result["lr"],
                    weight_decay=tuning_result["weight_decay"],
                    epochs=100,
                    created_on=date_time,
                    freq=tuning_result["freq"],
                    phase=tuning_result["phase"],
                    amp=tuning_result["amp"],
                    dist=tuning_result["dist"],
                    requires_grad=tuning_result["requires_grad"],
                    seed=seed,
                    subset=None,
                    index=i,
                    identifier=identifier))

            population_training.run()

except Exception as e:
    print(f"could not read: {e}")
