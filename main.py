import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from neuralnetwork import LinearNeuralNetwork, PopulationNeuralNetwork
from population_layer import PopulationCodedLayer
from trainer import Trainer

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

hidden_dim = 128

linearNetwork = LinearNeuralNetwork(28 * 28, hidden_dim, 10)
populationNetwork = PopulationNeuralNetwork(28 * 28, hidden_dim, 10)

for model in [populationNetwork]:
    print(f"Model: {model.name}")
    trainer = Trainer(
        model=model,
        training_data=training_data,
        test_data=test_data,
        batch_size=64
    )

    trainer.train(epochs=10)
    trainer.test()