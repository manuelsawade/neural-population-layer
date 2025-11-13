from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from torch import Tensor
import torch
import torch.nn as nn

from activations.neuron import NeuronPopulation, PreferredStimulus
from datasets.base import Dataset
from networks import NeuralNetwork
from populations import CircularPopulationBase, Distribution, MexicanHat, PopulationBase, SineWave, TuningCurve
from activations.sine_layer import PreferredValueActivation, PreferredValueInitializer
from activations.dynamic import SoftmaxGaussianActivation
from trainer import Trainer

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
    stack: str

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
        return "_".join([self.stack, self.dataset.name, str(self.training_noise), str(self.hidden_dim), str(self.test_noise), str(self.batch_size), str(self.learning_rate), str(self.weight_decay), str(self.epochs), "none" if self.subset is None else str(self.subset), str(self.index), self.identifier]).replace(" ", "").replace(",", "_").replace(".", "_")

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
class PreferredValueParameter(HyperParameter):
    freq: float | None
    phase: int | None
    amp: float | None
    dist: Distribution | None
    init: PreferredValueInitializer | None
    requires_grad: bool | None = False

    def toDict(self):
        dict = super().toDict()
        dict["freq"] = self.freq
        dict["phase"] = self.phase
        dict["amp"] = self.amp
        dict["dist"] = self.dist.name if self.dist else None
        dict["requires_grad"] = self.requires_grad
        return dict

    def get_output_file(self):
        return "_".join([super().get_output_file(), str(self.freq), str(self.phase), str(self.amp), str(self.dist), str(self.requires_grad)]).replace(" ", "").replace(",", "_").replace(".", "_")

@dataclass
class SoftmaxGaussianParameter(HyperParameter):
    activation: TuningCurve | MexicanHat 
    sigma: float 
    alpha: float
    normalize: bool

    def toDict(self):
        dict = super().toDict()
        dict["activation"] = self.activation
        dict["sigma"] = self.sigma
        dict["alpha"] = self.alpha
        dict["normalize"] = self.normalize
        return dict

    def get_output_file(self):
        return "_".join([super().get_output_file(), str(self.activation.name), str(self.sigma), str(self.alpha), str(self.normalize)]).replace(" ", "").replace(",", "_").replace(".", "_")   

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

class PreferredValueTraining(TrainingBase):
    hyper_parameter: PreferredValueParameter
    network = "preferred_population"

    def run(self):
        preference: SineWave | Tensor

        match self.hyper_parameter.init:
            case PreferredValueInitializer.SINE_WAVE:
                preference = SineWave(
                    size=self.hyper_parameter.hidden_dim,
                    freq=self.hyper_parameter.freq,
                    phase=self.hyper_parameter.phase,
                    amp=self.hyper_parameter.amp,
                    dist=self.hyper_parameter.dist,
                    requires_grad=self.hyper_parameter.requires_grad
                )
            case PreferredValueInitializer.RANDOM_NORMAL:
                preference = torch.randn(self.hyper_parameter.hidden_dim)
            case PreferredValueInitializer.RANDOM_UNIFORM:
                preference = torch.rand(self.hyper_parameter.hidden_dim)

        if not self.hyper_parameter.init == PreferredValueInitializer.SINE_WAVE:
            self.hyper_parameter.hidden_dim = None
            self.hyper_parameter.freq = None
            self.hyper_parameter.phase = None
            self.hyper_parameter.amp = None
            self.hyper_parameter.dist = None
            self.hyper_parameter.requires_grad = None

        stack = nn.Sequential(
            nn.Linear(self.hyper_parameter.dataset.input_dim, self.hyper_parameter.hidden_dim),
            PreferredValueActivation(initialized=preference), 
            nn.LazyLinear(self.hyper_parameter.dataset.output_dim),  
            )

        self.run_stack(stack)

class SoftmaxGaussianTraining(TrainingBase):
    hyper_parameter: SoftmaxGaussianParameter
    network = "softmax_gaussian"

    def run(self):
        stack = nn.Sequential(
            nn.Linear(self.hyper_parameter.dataset.input_dim, self.hyper_parameter.hidden_dim),
            SoftmaxGaussianActivation(
                activation=self.hyper_parameter.activation, 
                alpha=self.hyper_parameter.alpha, 
                sigma=self.hyper_parameter.sigma, 
                normalize=self.hyper_parameter.normalize), 
            nn.LazyLinear(self.hyper_parameter.dataset.output_dim))

        self.run_stack(stack)


class LinearNetworkTraining(TrainingBase):
    network = "linear"

    def run(self):
        stack = nn.Sequential(
                nn.Linear(self.hyper_parameter.dataset.input_dim, self.hyper_parameter.hidden_dim),
                nn.ReLU(), 
                nn.Linear(self.hyper_parameter.hidden_dim, self.hyper_parameter.dataset.output_dim))

        self.run_stack(stack)      