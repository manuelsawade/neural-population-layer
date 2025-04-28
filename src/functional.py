import torch

def norm_max(input):
    return input / input.max(dim=-1, keepdim=True).values