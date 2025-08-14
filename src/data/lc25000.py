import math
import os
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from data.noise import AddGaussianNoise

class LC25000():
    def __init__(self, size=768*768):
        self.size = size

    def __call__(self, training_noise=0.0):
        transform_with_noise = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0.0, training_noise),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

        training_data = LC25000Dataset(
            root="data/lc25000",
            size=self.size,
            train=True,
            transform=transform_with_noise
        )

        test_data = LC25000Dataset(
            root="data/lc25000",
            size=self.size,
            train=False,
            transform=ToTensor()
        )

        return training_data, test_data

class LC25000Dataset(Dataset):
    def __init__(self, root,size=768*768, train=True, transform=None, seed=42):
        self.root_dir = root
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.class_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.size = size

        self.image_paths = []
        self.labels = []

        for cls in self.classes:
            cls_dir = os.path.join(root, cls)

            image_paths = []
            labels = []
            
            if os.path.isdir(cls_dir):
                for img_file in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_file)
                    
                    if os.path.isfile(img_path):
                        image_paths.append(img_path)
                        labels.append(self.class_idx[cls])
                   
            gen = torch.Generator().manual_seed(seed)
            train_paths, test_paths = torch.utils.data.random_split(image_paths, [0.8, 0.2], generator=gen)
            train_labels, test_labels = torch.utils.data.random_split(labels, [0.8, 0.2], generator=gen)

            if train:
                self.image_paths.extend(train_paths)
                self.labels.extend(train_labels)
            else:
                self.image_paths.extend(test_paths)
                self.labels.extend(test_labels)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('L')
        image_len = int(math.sqrt(self.size))
        image = image.resize((image_len, image_len))

        if self.transform:
            image = self.transform(image)

        return image, label
        