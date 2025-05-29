from datetime import datetime
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from matplotlib import pyplot as plt

class Trainer:
    def __init__(self, model, training_data, test_data, batch_size=64, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs):
        self.model.to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            for x, y in self.train_loader:
                pred = self.model(x)
                loss = self.loss_fn(pred, y)

                loss.backward()
                self.optimizer.step()
                for param in self.model.parameters():
                    param.grad = None

            sys.stdout.write(f'\rEpoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            sys.stdout.flush()
            

    def test(self, noise=0.01, summary={}):
        self.model.eval()
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, correct = 0, 0

        activations = {}

        def get_activation(name, type):
            def hook(module, input, output):                
                if name not in activations:
                    activations[name] = {}
                    activations[name]['type'] = type
                    activations[name]['out'] = []
                    activations[name]['mean'] = 0
                    activations[name]['std'] = []
                    activations[name]['len'] = []
                
                activations[name]['out'].append(output.detach().flatten(start_dim=1).tolist())
                activations[name]['mean'] += output.detach().mean()
                activations[name]['std'].append(output.detach().std())
                activations[name]['len'].append(len(output.detach()))
            return hook
        
        for index in range(len(self.model.layers)):
            self.model.layers[index].register_forward_hook(get_activation(f'layers.{index}', self.model.layers[index]._get_name()))

        sharpness_scores = {}

        with torch.no_grad():
            for x, y in self.test_loader:
                pred = self.model(x)

                base_loss = self.loss_fn(pred, y).item()
                test_loss += base_loss

                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                for name, param in self.model.named_parameters():
                    if param.requires_grad == False: continue

                    if 'phase' in name: continue

                    original = param.data.clone()
                    
                    param_noise = noise * torch.randn_like(param)
                    param.data += param_noise

                    perturbed_loss = self.loss_fn(self.model(x), y).item()

                    if name not in sharpness_scores:
                        sharpness_scores[name] = []

                    sharpness_scores[name].append(perturbed_loss - base_loss)

                    param.data = original
        


        test_loss /= num_batches
        correct /= size

        print(f"\nTest \n  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>4f} ")
        print(f"Activations")
        for layer in activations:
            stds = torch.tensor(activations[layer]['std'])
            sizes = torch.tensor(activations[layer]['len'])

            variances = stds ** 2
            weighted = torch.sum(variances * sizes) / torch.sum(sizes)
            std = torch.sqrt(weighted)

            print(f"  {layer}: {activations[layer]['mean'] / num_batches:>4f} (mean), {std:>4f} (std)")

        print(f"Sharpness")
        for param in sharpness_scores:
            print(f"  {param}: {sum(sharpness_scores[param]) / len(sharpness_scores[param]):>4f}")

        fig, axs = plt.subplots(len(activations), figsize=(12, 10))

        for index, layer in enumerate(activations):
            axs[index].set_title(f'{index}: {activations[layer]['type']}')
            axs[index].set_xlabel('Neuron')
            axs[index].set_ylabel('Layer')

            flat_out = []
            for batch in activations[layer]['out']:
                flat_out.append(batch[:self.batch_size][0])

            im = axs[index].imshow(flat_out[:20], cmap='inferno', interpolation='nearest')

            fig.colorbar(im, ax=axs[index])

        font_size = 14

        summary['Accuracy'] = 100*correct
        summary['Avg. loss'] = test_loss
        summary['Batch size'] = self.batch_size

        for index, param in enumerate(summary):
            axs[-1].text(-65, index * 2, f'{param}: {summary[param]}', fontsize=font_size)

        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        plt.savefig(f'./images/result_{date_time}.png')

        print("\n")