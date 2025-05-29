from matplotlib import pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from data.noise import AddGaussianNoise

class MNIST():
    def __call__(self, training_noise=0.0):
        transform_with_noise = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0.0, training_noise),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform_with_noise
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        return training_data, test_data