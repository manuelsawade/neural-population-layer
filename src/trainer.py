from collections import OrderedDict
from datetime import datetime
from functools import partial
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch import Tensor
import os
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import random_split
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
from ray.tune.schedulers import ASHAScheduler

from metrics.activations import activation_metric
from metrics.noise_sensitivity import noise_sensitivity_metric
from metrics.roby import roby_metric
from metrics.sharpness import sharpness_metric

class Trainer:
    def __init__(self, 
                 model: nn.Sequential, 
                 dataset, 
                 subset: int | None = None,
                 batch_size=64, 
                 learning_rate=0.001, 
                 training_noise=0.2,
                 weight_decay=0.0001):
        training_data, test_data = dataset(training_noise=training_noise)

        if subset:
            training_data = Subset(training_data, list(range(subset)))
            test_data = Subset(test_data, list(range(int(subset / 10))))

        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
        self.training_data = training_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.flatten = nn.Flatten()  
    
    def train(self, epochs):       
        self.epochs = epochs
        self.model.to(self.device)

        loss_sum = 0

        for epoch in range(epochs):
            start = time.time()
            self.model.train()

            roby_scores: dict[str, float] = {}

            count = 1
            correct, total = 0, 0
            for x, y in self.train_loader:                
                pred: Tensor = self.model(x)

                correct += (pred.argmax(dim=1) == y).sum().item()

                loss: Tensor = self.loss_fn(pred, y)
                loss.backward()

                self.optimizer.step()
                for param in self.model.parameters():
                    param.grad = None

                loss_sum += loss

                sys.stdout.write(f'\rEpoch [{epoch+1:02}/{epochs}], Batch [{count * self.batch_size}]')
                sys.stdout.flush()

                roby_metric(x, pred, p=float('inf'), metric=['fsa'],
                    append_to=roby_scores)
                
                roby_metric(x, pred, p=2, metric=['fsa'],
                    append_to=roby_scores)
                
                count += 1
                total += y.size(0)

            end = time.time()
            if correct != 0:
                acc = correct / total
            else: 
                acc = 0

            loss_sum /= len(self.train_loader)
            fsa_2 = sum(roby_scores["fsa_2"]) / len(roby_scores["fsa_2"])
            fsa_inf = sum(roby_scores["fsa_inf"]) / len(roby_scores["fsa_inf"])

            sys.stdout.write(f'\rEpoch [{epoch+1:02}/{epochs}], Loss: {loss_sum:.4f}, Acc: {acc*100:.4f}, FSA_2: {fsa_2:.4f}, FSA_inf: {fsa_inf:.4f}, Elapsed Time: {end - start:.2f}s')
            sys.stdout.write('\n')

    def test(self, noise=0.01, summary: dict[str, object]={}, write_file=True):
        #path, file_name = self._create_summary(noise, summary)

        self.model.eval()
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, correct = 0, 0

        activation_scores: dict[str, dict[str, object]] = {}
        sharpness_scores: dict[str, float] = {}
        roby_scores: dict[str, float] = {}
        noise_sensitivity_scores: dict[str, list[float]] = {}

        activation_metric(self.model, append_to=activation_scores)

        with torch.no_grad():
            for x, y in self.test_loader:
                pred = self.model(x)

                base_loss = self.loss_fn(pred, y).item()
                test_loss += base_loss

                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                sharpness_metric(self.model, x, y, noise, base_loss, self.loss_fn, 
                    append_to=sharpness_scores)
                
                roby_metric(x, pred, p=float('inf'), metric=['fsa', 'fsd'],
                    append_to=roby_scores)
                
                roby_metric(x, pred, p=2, metric=['fsa', 'fsd'],
                    append_to=roby_scores)
                
                noise_sensitivity_metric(self.model, x, y, attack='fgsm', topk=self.dataset.output_dim, 
                    append_to=noise_sensitivity_scores)
                
                #autoattack_metric(self.model, x, y, self.device, eps=8/255, norm='Linf', log_path=path)
        
        test_loss /= num_batches
        correct /= size

        activation_output = {}

        #print(f"\nTest \n  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>4f} ")
        print(f"Activations")
        for layer in activation_scores['scores']:
            stds = torch.tensor(activation_scores['scores'][layer]['std'])
            sizes = torch.tensor(activation_scores['scores'][layer]['len'])

            variances = stds ** 2
            weighted = torch.sum(variances * sizes) / torch.sum(sizes)
            std = torch.sqrt(weighted)

            activation_output[layer] = {}
            activation_output[layer]['std'] = std.item()
            activation_output[layer]['mean'] = (activation_scores['scores'][layer]['mean'] / num_batches).item()

            print(f"  {layer}: {activation_scores['scores'][layer]['mean'] / num_batches:>4f} (mean), {std:>4f} (std)")

        print(f"Sharpness")
        sharpness_output = {}
        for param in sharpness_scores:
            sharpness_output[param] = sum(sharpness_scores[param]) / len(sharpness_scores[param])
            print(f"  {param}: {sum(sharpness_scores[param]) / len(sharpness_scores[param]):>4f}")
        
        print(f"Roby")
        roby_output = {}
        roby_scores = OrderedDict(sorted(roby_scores.items()))
        for param in roby_scores:
            mean = sum(roby_scores[param]) / len(roby_scores[param])
            std = torch.std(torch.tensor(roby_scores[param]))

            roby_output[param] = {}
            roby_output[param]['mean'] = sum(roby_scores[param]) / len(roby_scores[param])
            roby_output[param]['std'] = std.item()

            print(f"  {param}: {mean:>4f} (mean)")
            print(f"  {param}: {std:>4f} (std)")

        print(f"Noise Sensitivity")
        noise_sensitivity_output = {}
        for param in noise_sensitivity_scores:
            mean = sum(noise_sensitivity_scores[param]) / len(noise_sensitivity_scores[param])
            std = torch.std(torch.tensor(noise_sensitivity_scores[param]))
            
            noise_sensitivity_output[param] = {}
            noise_sensitivity_output[param]['mean'] = sum(noise_sensitivity_scores[param]) / len(noise_sensitivity_scores[param])
            noise_sensitivity_output[param]['std'] = std.item()

            print(f"  {param}: {mean:>4f} (mean)")
            print(f"  {param}: {std:>4f} (std)")

        #if write_file:
            # fig, axs = plt.subplots(len(activation_scores['output']), figsize=(40, 20), sharey=True)

            # for index, layer in enumerate(activation_scores['output']):
            #     axs[index].set_title(f'{index}: {activation_scores['output'][layer]['type']}')
            #     axs[index].set_xlabel('Neuron')
            #     axs[index].set_ylabel('Layer')            

            #     im = axs[index].imshow(activation_scores['output'][layer]['values'][:20], cmap='inferno', interpolation='nearest')

            #     fig.colorbar(im, ax=axs[index])

        summary['avg_loss'] = test_loss
        summary['accuracy'] = 100*correct
        summary['sharpness_scores'] = sharpness_output
        summary['activation_scores'] = activation_output
        summary['ruby_scores'] = roby_output
        summary['noise_sensitivity'] = noise_sensitivity_output

            #plt.savefig(f'{file_name}.png')
            # with open(f'{file_name}.json', mode='w', newline='') as file:
            #     file.write(json.dumps(summary, indent=4))

        print("\n")

    def _create_summary(self, noise: float, summary: dict[str, object]={}) -> str:
        summary['epochs'] = self.epochs
        summary['batch_size'] = self.batch_size
        summary['dataset'] = self.dataset.name
        summary['test_noise'] = noise
        
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

        path = f'./results/{summary['dataset']}_{summary['network']}/{summary['stack']}/{date_time}/'
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = f'{path}/result_{date_time}'

        return path, file_name