import torch

class Gaussian:
    def mask(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (2 * sigma ** 2)
    
class MexicanHat:
    def mask(self, x, mu, sigma):
        return (1 - ((x - mu) / sigma) ** 2) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
class Oscillatory:
    def mask(self, x, mu, sigma):
        positions = torch.arange(x, device=x.device).float()

        theta = 2 * torch.pi * self.freq * positions / x + self.phase
        sine_mask = (torch.sin(theta) + 1) / 2
        