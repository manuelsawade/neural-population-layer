from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor

class DecoderBase(nn.Module):
    name: str = ""

    def __init__(self):
        super().__init__()

    def forward(self, out: Tensor, mu: Tensor, orientation: tuple[float, float]) -> Tensor:
        pass

class WeightedAverageDecoder(DecoderBase):
    name = "weighted_average"

    def forward(self, out: Tensor, mu: Tensor, orientation: tuple[float, float]) -> Tensor:
        denom = out.sum(dim=-1) + 1e-8
        cont = (out * mu).sum(dim=-1, keepdim=False) / denom
        rounded = cont.clamp(orientation[0], orientation[1])

        return rounded
    
class CircularMeanDecoder(DecoderBase):
    name = "circular_mean"

    def forward(self, out: Tensor, mu: Tensor, orientation: tuple[float, float])-> Tensor:
        L = orientation[1] - orientation[0]
        theta = 2 * torch.pi * (mu - orientation[0]) / L

        C = torch.sum(out * torch.cos(theta), dim=-1) 
        S = torch.sum(out * torch.sin(theta), dim=-1) 

        eps = 1e-8
        theta_hat = torch.atan2(S + eps, C + eps)

        decoded = orientation[0] + (L / (2 * torch.pi)) * theta_hat
        return decoded