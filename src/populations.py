import torch

class Gaussian:
    def mask(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (2 * sigma ** 2)
    
class MexicanHat:
    def mask(self, x, mu, sigma):
        return (1 - ((x - mu) / sigma) ** 2) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
        