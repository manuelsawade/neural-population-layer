from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Generic, Iterator, TypeVar
import torch
import torch.nn as nn

from activations.neuron import NeuronPopulation, PreferredStimulus, WeightedAverageDecoder
from datasets.base import Dataset
from networks import NeuralNetwork
from populations import CircularPopulationBase, CircularTuningCurve, PopulationBase, TuningCurve
from trainer import Trainer
from datasets.mnist import MNIST

@dataclass
class HyperParameter:
    dataset: Dataset | None
    hidden_dim: int
    training_noise: float
    test_noise: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    created_on: datetime
    seed: int
    subset: int | None
    index: int
    identifier: str

    def toDict(self):
        return {
            "dataset": self.dataset.name,
            "hidden_dim": self.hidden_dim,
            "training_noise": self.training_noise,
            "test_noise": self.test_noise,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "created_on": self._dateToString(self.created_on),
            "seed": self.seed,
            "subset": self.subset
        }

    def get_output_folder(self, base: str = './experiments'):
        return f'{base}/{self.identifier}/'
    
    def _dateToString(self, datetime: datetime) -> str:
        return datetime.strftime("%Y_%m_%d_%H_%M_%S")
    
    def get_output_file(self):
        return "_".join([self.dataset.name, str(self.hidden_dim), str(self.training_noise), str(self.test_noise), str(self.batch_size), str(self.learning_rate), str(self.weight_decay), str(self.epochs), "none" if self.subset is None else str(self.subset), str(self.index), self.identifier]).replace(" ", "").replace(",", "_").replace(".", "_")

@dataclass
class NeuronPopulationParameter(HyperParameter):
    sigma: float
    neurons: int
    orientation: tuple[float, float]
    activation: PopulationBase | CircularPopulationBase
    stimulus: PreferredStimulus

    def toDict(self):
        dict = super().toDict()
        dict["sigma"] = self.sigma
        dict["neurons"] = self.neurons
        dict["orientation"] = self.orientation
        dict["activation"] = self.activation.name
        dict["stimulus"] = self.stimulus.name

        return dict
    
    def get_output_file(self):
        return "_".join([super().get_output_file(), str(self.sigma), str(self.neurons), str(self.orientation), self.activation.name, str(self.stimulus)]).replace(" ", "").replace(",", "_").replace(".", "_")
        

@dataclass
class TrainingBase:
    hyper_parameter: HyperParameter
    network: str = ""

    def __init_subclass__(cls):
        cls.output: dict = {}
        cls.output['network'] = cls.network

    def run_stack(self, stack: nn.Sequential):           
        folder_path = self.hyper_parameter.get_output_folder()
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = f'{folder_path}/{self.hyper_parameter.get_output_file()}.json'
        if Path(file_path).exists():
            print(f"Skipped training: File already exists: {file_path}")
            return
                
        trainer = Trainer(
            model=NeuralNetwork(layers=stack),
            dataset=self.hyper_parameter.dataset,
            training_noise=self.hyper_parameter.training_noise,
            batch_size=self.hyper_parameter.batch_size,
            learning_rate=self.hyper_parameter.learning_rate,
            weight_decay=self.hyper_parameter.weight_decay,
            subset=self.hyper_parameter.subset
        )

        trainer.train(epochs=self.hyper_parameter.epochs, output=file_path)
        trainer.test(noise=self.hyper_parameter.test_noise, summary=self.output, write_file=False)

        self.output['hyper_parameter'] = self.hyper_parameter.toDict()

        with open(file_path, mode='w', newline='') as file:
            file.write(json.dumps(self.output, indent=4))

class NeuronPopulationTraining(TrainingBase):
    hyper_parameter: NeuronPopulationParameter
    network = "population"

    def run(self):
        stack = nn.Sequential(
            nn.Linear(self.hyper_parameter.dataset.input_dim, self.hyper_parameter.hidden_dim),
            NeuronPopulation(
                self.hyper_parameter.hidden_dim, 
                sigma=self.hyper_parameter.sigma,
                neurons=self.hyper_parameter.neurons,
                orientation=self.hyper_parameter.orientation,
                activation=self.hyper_parameter.activation,
                stimulus=self.hyper_parameter.stimulus),
            nn.LazyLinear(self.hyper_parameter.dataset.output_dim),     
            )

        self.run_stack(stack) 


class LinearNetworkTraining(TrainingBase):
    network = "linear"

    def run(self):
        stack = nn.Sequential(
                nn.Linear(self.hyper_parameter.dataset.input_dim, self.hyper_parameter.hidden_dim),
                nn.ReLU(), 
                nn.Linear(self.hyper_parameter.hidden_dim, self.hyper_parameter.dataset.output_dim))

        self.run_stack(stack)      





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
