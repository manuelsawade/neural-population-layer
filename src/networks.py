import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super().__init__()  
        self.name = __class__.__name__
        self.flatten = nn.Flatten()
        self.layers = layers
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits
    
class NeuralPopNetwork(nn.Module):
    def __init__(self, layers):
        super().__init__()  
        self.name = __class__.__name__
        self.flatten = nn.Flatten()
        self.layers = layers
        
    def forward(self, x):
        x = self.flatten(x)
        out_pop = self.layers(x)
        return out_pop