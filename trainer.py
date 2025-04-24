import torch
import torch.nn as nn
from torch.utils.data import DataLoader 

class Trainer:
    def __init__(self, model, training_data, test_data, batch_size=64):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters())
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for x, y in self.train_loader:
                pred = self.model(x)
                loss = self.loss_fn(pred, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def test(self):
        self.model.eval()
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                pred = self.model(x)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")