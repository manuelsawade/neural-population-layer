from matplotlib import pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import ssl

from datasets.noise import AddGaussianNoise

class CIFAR10():    
    input_dim = 32 * 32 * 3
    output_dim = 10
    name = "cifar10"

    def __call__(self, training_noise=0.0):
        transform_with_noise = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0.0, training_noise),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

        ssl._create_default_https_context = ssl._create_unverified_context  

        training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform_with_noise
        )

        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )


        return training_data, test_data