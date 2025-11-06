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

    def __call__(self, training_noise=0.0, noise_probability=1.0):
        def clamp_transform(tensor):
            return torch.clamp(tensor, 0.0, 1.0)
        
        def add_noise(tensor):
            if noise_probability <= 0.0:
                return tensor

            if training_noise < torch.rand(1).item() or training_noise >= 1.0:  
                noise = torch.randn_like(tensor) * training_noise + 0.0
                return tensor + noise
            
            return tensor
        
        transform_with_noise = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(add_noise),
            transforms.Lambda(clamp_transform)
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