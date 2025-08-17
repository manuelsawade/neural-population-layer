from enum import Enum
import torch

class Gaussian:
    def __call__(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (2 * sigma ** 2)
    
class TuningCurve:
    def __call__(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
class MexicanHat:
    def __call__(self, x, mu, sigma):
        return (1 - ((x - mu) / sigma) ** 2) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
class LogNormal:
    def __call__(self, x, mu, sigma):
        x = torch.clamp(x, min=1e-10)
        coeff = 1.0 / (x * sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
        exponent = -((torch.log(x) - mu) ** 2) / (2 * sigma**2)
        return coeff * torch.exp(exponent)

class Distribution(Enum):
    ZERO_MEAN = 0,
    ZERO_BASE = 1,

class SineWave:
    def __call__(self, freq, phase, amp, x_pos, x_size, dist=Distribution.ZERO_MEAN):
        if dist == Distribution.ZERO_MEAN: 
            return self._sine(freq, phase, amp, x_pos, x_size)
        
        return self._sine_base(freq, phase, amp, x_pos, x_size)
        

    def _sine(self, freq, phase, amp, x_pos, x_size): 
        return amp * torch.sin(2 * torch.pi * freq * x_pos / x_size + phase)

    def _sine_base(self, freq, phase, amp, x_pos, x_size):
        return amp * (torch.sin(2 * torch.pi * freq * x_pos / x_size + phase) + 1)
        