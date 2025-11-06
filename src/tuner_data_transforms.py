#!/usr/bin/python3

import torch

training_noise=1.0
training_noise_probability=0.5

def clamp_transform(tensor):
    return torch.clamp(tensor, 0.0, 1.0)

def add_noise(tensor):
    if training_noise_probability <= 0.0:
        return tensor
    
    if training_noise_probability < torch.rand(1).item() or training_noise_probability >= 1.0:
        noise = torch.randn_like(tensor) * training_noise + 0.0
        return tensor + noise
    
    return tensor