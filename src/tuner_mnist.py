from collections import OrderedDict
from datetime import datetime
from functools import partial
import random
from typing import Any
import torch
import torch.nn as nn
import os
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import random_split
from ray import tune
from ray import train
from ray.tune import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
from ray.tune.schedulers import ASHAScheduler

from metrics.roby import roby_metric
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import ssl
import torchvision

from datasets.noise import AddGaussianNoise
from activations.neuron import NeuronPopulation, PreferredStimulus
from datasets.cifar10 import CIFAR10
from decoder import WeightedAverageDecoder
from populations import TuningCurve
from metrics.activations import activation_metric
from metrics.noise_sensitivity import noise_sensitivity_metric
from metrics.sharpness import sharpness_metric
from training import training

# Best trial config: {'lr': 0.0001711898326456389, 'batch_size': 128, 'hidden_dim': 200, 'sigma': 0.6, 'neurons': 12, 'orientation': (-4, 4), 'stimulus': <PreferredStimulus.RAND_NORMAL: (0,)>}
# Best trial final validation loss: 1.7526937300645853
# Best trial final validation accuracy: 0.3828
# Best trial final validation fsa_inf: 0.9675893527043017
# (func pid=497) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/Users/manuelsawade/ray_results/run_2025-11-02_21-44-18/run_b39ec_00004_4_batch_size=128,hidden_dim=400,lr=0.0432,neurons=12,orientation=-4_4,sigma=0.8000,stimulus=ref_ph_e8394d2e_2025-11-02_21-44-18/checkpoint_000009)

# Test 
#   Accuracy: 39.0%, Avg loss: 1.742967 
# Roby
#   fsa_2: 0.476573 (mean)
#   fsa_2: 0.095572 (std)
#   fsa_inf: 0.497749 (mean)
#   fsa_inf: 0.088072 (std)
#   fsd_2: 0.423995 (mean)
#   fsd_2: 0.041864 (std)
#   fsd_inf: 0.473770 (mean)
#   fsd_inf: 0.044186 (std)

date_time = datetime.now()

identifier = "cifar10_v2_0"
input_dim = 32 * 32 * 3
output_dim = 10

seed = random.randint(1000000, 9999999)
torch.manual_seed(seed)  

def clamp_transform(tensor):
    return torch.clamp(tensor, 0.0, 1.0)

def add_noise(tensor):
    noise = torch.randn_like(tensor) * 0.1 + 0.0
    return tensor + noise

def load_data(): 
    ssl._create_default_https_context = ssl._create_unverified_context 

    training_data = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data

def run(config, data_dir=None):
        stack = nn.Sequential(
            nn.Linear(input_dim, config["hidden_dim"]),
            NeuronPopulation(
                config["hidden_dim"], 
                sigma=config["sigma"],#sigma=hyper_parameter.sigma,
                neurons=config["neurons"],#neurons=hyper_parameter.neurons,
                orientation=config["orientation"],#orientation=hyper_parameter.orientation,
                activation=TuningCurve(readout=WeightedAverageDecoder()),
                stimulus=config["stimulus"]),
            nn.LazyLinear(output_dim),     
            )

        optimizer = torch.optim.AdamW(
            stack.parameters(), 
            lr=config["lr"])
        
        loss_fn = nn.CrossEntropyLoss()

        #optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)
                start_epoch = checkpoint_state["epoch"]
                stack.load_state_dict(checkpoint_state["net_state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        else:
            start_epoch = 0

        training_data, test_data = load_data()

        test_abs = int(len(training_data) * 0.8)
        train_subset, val_subset = random_split(
            training_data, [test_abs, len(training_data) - test_abs]
        )

        trainloader = torch.utils.data.DataLoader(
            train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=1
        )
        valloader = torch.utils.data.DataLoader(
            val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=1
        )

        flatten = nn.Flatten()

        test_fsa_scores = {}

        for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                flatten = nn.Flatten()

                inputs = flatten(inputs)

                outputs = stack(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics

                roby_metric(outputs, inputs, p=float('inf'), metric=['fsa'],
                    append_to=test_fsa_scores)

                running_loss += loss.item()
                epoch_steps += 1
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f, fsa_inf: %.3f"
                        % (epoch + 1, i + 1, running_loss / epoch_steps, sum(test_fsa_scores["fsa_inf"]) / len(test_fsa_scores["fsa_inf"]))
                    )
                    running_loss = 0.0

            # Validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0

            val_fsa_scores = {}

            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data

                    inputs = flatten(inputs)

                    outputs = stack(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = loss_fn(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

                    roby_metric(outputs, inputs, p=float('inf'), metric=['fsa'],
                        append_to=val_fsa_scores)

            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": stack.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                tune.report(
                    {
                        "loss": val_loss / val_steps, 
                        "accuracy": correct / total, 
                        "fsa_inf": sum(val_fsa_scores["fsa_inf"]) / len(val_fsa_scores["fsa_inf"])
                    },
                    checkpoint=checkpoint,
                )

        print("Finished Training") 

def test(config, checkpoint_data, noise=0.2):
    model = nn.Sequential(
        nn.Linear(input_dim, config["hidden_dim"]),
        NeuronPopulation(
            config["hidden_dim"], 
            sigma=config["sigma"],#sigma=hyper_parameter.sigma,
            neurons=config["neurons"],#neurons=hyper_parameter.neurons,
            orientation=config["orientation"],#orientation=hyper_parameter.orientation,
            activation=TuningCurve(readout=WeightedAverageDecoder()),
            stimulus=config["stimulus"]),
        nn.LazyLinear(output_dim),     
            )   
    
    model.load_state_dict(checkpoint_data["net_state_dict"])

    model.eval()

    training_data, test_data = load_data()
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=int(config["batch_size"]), shuffle=False, num_workers=1
    )

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    sharpness_scores: dict[str, float] = {}
    roby_scores: dict[str, float] = {}
    noise_sensitivity_scores: dict[str, list[float]] = {}

    #activation_metric(model, append_to=activation_scores)

    flatten = nn.Flatten()

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in test_loader:
            x = flatten(x)
            pred = model(x)

            base_loss = loss_fn(pred, y).item()
            test_loss += base_loss

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # sharpness_metric(model, x, y, noise, base_loss, loss_fn, 
            #     append_to=sharpness_scores)
            
            roby_metric(x, pred, p=float('inf'), metric=['fsa', 'fsd'],
                append_to=roby_scores)
            
            roby_metric(x, pred, p=2, metric=['fsa', 'fsd'],
                append_to=roby_scores)
            
            # noise_sensitivity_metric(model, x, y, attack='fgsm', topk=output_dim, 
            #     append_to=noise_sensitivity_scores)
            
            #autoattack_metric(self.model, x, y, self.device, eps=8/255, norm='Linf', log_path=path)
    
    test_loss /= num_batches
    correct /= size

    print(f"\nTest \n  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>4f} ")
    # print(f"Activations")
    # for layer in activation_scores['scores']:
    #     stds = torch.tensor(activation_scores['scores'][layer]['std'])
    #     sizes = torch.tensor(activation_scores['scores'][layer]['len'])

    #     variances = stds ** 2
    #     weighted = torch.sum(variances * sizes) / torch.sum(sizes)
    #     std = torch.sqrt(weighted)

    #     activation_output[layer] = {}
    #     activation_output[layer]['std'] = std.item()
    #     activation_output[layer]['mean'] = (activation_scores['scores'][layer]['mean'] / num_batches).item()

    #     print(f"  {layer}: {activation_scores['scores'][layer]['mean'] / num_batches:>4f} (mean), {std:>4f} (std)")

    # print(f"Sharpness")
    # sharpness_output = {}
    # for param in sharpness_scores:
    #     sharpness_output[param] = sum(sharpness_scores[param]) / len(sharpness_scores[param])
    #     print(f"  {param}: {sum(sharpness_scores[param]) / len(sharpness_scores[param]):>4f}")
    
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

    # print(f"Noise Sensitivity")
    # noise_sensitivity_output = {}
    # for param in noise_sensitivity_scores:
    #     mean = sum(noise_sensitivity_scores[param]) / len(noise_sensitivity_scores[param])
    #     std = torch.std(torch.tensor(noise_sensitivity_scores[param]))
        
    #     noise_sensitivity_output[param] = {}
    #     noise_sensitivity_output[param]['mean'] = sum(noise_sensitivity_scores[param]) / len(noise_sensitivity_scores[param])
    #     noise_sensitivity_output[param]['std'] = std.item()

    #     print(f"  {param}: {mean:>4f} (mean)")
    #     print(f"  {param}: {std:>4f} (std)")

    print("\n")

def main(data_dir):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([128, 256]),
        "hidden_dim": tune.choice([200, 400, 600]),
        "sigma": tune.choice([0.6, 0.8, 1.2]),
        "neurons": tune.choice([8, 12, 16]),
        "orientation": tune.choice([(-4,4)]),
        "stimulus": tune.choice([PreferredStimulus.LINEAR, PreferredStimulus.RAND_NORMAL]),
    }

    load_data()

    scheduler = ASHAScheduler(
        metric="fsa_inf",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2,
    ) 

    result: Any | None

    try:
        result = tune.run(
            partial(run, data_dir=data_dir),
            config=config,
            num_samples=10,
            scheduler=scheduler
        )
    except:
        print("error")

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    print(f"Best trial final validation fsa_inf: {best_trial.last_result['fsa_inf']}")

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        test(best_trial.config, best_checkpoint_data)   

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(identifier)