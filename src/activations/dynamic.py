from enum import StrEnum
import torch
import torch.linalg as L
import torch.nn as nn
import torch.nn.functional as F

from populations import MexicanHat, TuningCurve

class Gaussian(StrEnum):
    TuningCurve = "tuning_curve",
    MexicanHat = "mexican_hat"

class SoftmaxGaussianActivation(nn.Module):
    def __init__(self, activation: MexicanHat | TuningCurve, alpha=10.0, sigma=0.2, normalize=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.activation = activation
        self.normalize = normalize

    def forward(self, x):
        if self.normalize:
            x = F.normalize(x, p=float("inf"))

        p = F.softmax(self.alpha * x, dim=1)      
        output = self.activation.activation(x=x, mu=p, sigma=self.sigma)
        
        return output 