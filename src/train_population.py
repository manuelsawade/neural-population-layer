
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import random

import torch

from activations.neuron import PreferredStimulus
from datasets.mnist import MNIST
from decoder import CircularMeanDecoder, WeightedAverageDecoder
from populations import CircularTuningCurve, TuningCurve
from datasets.cifar10 import CIFAR10
from training.training import NeuronPopulationParameter, NeuronPopulationTraining

date_time = datetime.now()

identifier = "mnist_evaluation_population_encoding"
path = f"./experiments/{identifier}/tuning/"

folder_path = Path(path)
file_paths = sorted(folder_path.glob("population*.json"))
    
def run(p: tuple):
    print("load file:", p)
    try:
        with open(p, "r") as f:
            tuning_result = json.loads(f.read())

        for i in range(1):
            seed = random.randint(1000000, 9999999)
            torch.manual_seed(seed)  
            print(tuning_result)
            population_training = NeuronPopulationTraining(
                hyper_parameter=NeuronPopulationParameter(
                    stack="population_encoding",
                    dataset=MNIST(),
                    hidden_dim=tuning_result["hidden_dim"],
                    training_noise=tuning_result["noise"],
                    test_noise=0.2,
                    batch_size=tuning_result["batch_size"],
                    learning_rate=tuning_result["lr"],
                    weight_decay=tuning_result["weight_decay"],
                    epochs=100,
                    created_on=date_time,
                    sigma=tuning_result["sigma"],
                    orientation=(tuning_result["orientation"][0], tuning_result["orientation"][1]),
                    stimulus=PreferredStimulus(tuning_result["stimulus"]),
                    neurons=tuning_result["neurons"],
                    seed=seed,
                    subset=None,
                    activation=TuningCurve(readout=WeightedAverageDecoder()),
                    index=i,
                    identifier=identifier,
                    encoded_output=True))

            population_training.run()

    except Exception as e:
        print(f"could not read: {e}")

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=1) as executor:
        list(executor.map(run, file_paths))
