import torch.nn as nn

from training import MNISTTraining

class Lin1Lin2Lin3(MNISTTraining):
    def __init__(self, hidden_dim, epochs, batch_size, train_noise, test_noise, **kwargs):
        super().__init__(epochs, batch_size, train_noise, test_noise, kwargs)
        self.hidden_dim = hidden_dim
        
    def run(self):
        self.train_stack(nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.output_dim)))

