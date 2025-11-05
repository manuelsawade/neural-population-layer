#!/usr/bin/python3

from collections import OrderedDict
from datetime import datetime
from functools import partial
import json
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

# population

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

# Best trial config: {'lr': 0.0006032870111961857, 'batch_size': 256, 'hidden_dim': 200, 'sigma': 0.8, 'neurons': 12, 'orientation': (-4, 4), 'stimulus': <PreferredStimulus.RAND_NORMAL: (0,)>}                                                                                                                   
# neural-population-layer-1  | Best trial final validation loss: 1.496772050857544
# neural-population-layer-1  | Best trial final validation accuracy: 0.4668                                                                                                                                                                                                                                                                     
# neural-population-layer-1  | Best trial final validation fsa_inf: 0.9093379631638527                                                                                                                                                                                                                                                          
# neural-population-layer-1  | 
# neural-population-layer-1  | Test 
# neural-population-layer-1  |   Accuracy: 48.3%, Avg loss: 1.466806 

# neural-population-layer-1  | Stack: population, Noise: 1.0
# neural-population-layer-1  | Best trial config: {'lr': 0.0016917007678706566, 'batch_size': 256, 'hidden_dim': 400, 'sigma': 1.2, 'neurons': 12, 'orientation': (-4, 4), 'stimulus': <PreferredStimulus.RAND_NORMAL: (0,)>}                                                                                                                                 
# neural-population-layer-1  | Best trial final validation loss: 1.917538046836853
# neural-population-layer-1  | Best trial final validation accuracy: 0.3184
# neural-population-layer-1  | Best trial final validation fsa_inf: 0.4068716123700142                                                                                          
# neural-population-layer-1  |                                                                                                                                                  
# neural-population-layer-1  | Test 
# neural-population-layer-1  |   Accuracy: 36.3%, Avg loss: 1.864114                                                                                                            
# neural-population-layer-1  | Roby                                                                                                                                             
# neural-population-layer-1  |   fsa_2: 0.461510 (mean)                                                                                                                         
# neural-population-layer-1  |   fsa_2: 0.093098 (std)


# linear
# neural-population-layer-1  | Best trial config: {'lr': 0.0005609176336288977, 'batch_size': 64, 'hidden_dim': 200}                                                                                                                                                                                                                            
# neural-population-layer-1  | Best trial final validation loss: 1.9180927276611328
# neural-population-layer-1  | Best trial final validation accuracy: 0.3044                                                                                                                                                                                                                                                                    
# neural-population-layer-1  | Stack: linear, Noise: 1.0                                                                                                                                                                                                                                                                                        
# neural-population-layer-1  | Best trial config: {'lr': 0.0001243228025869031, 'batch_size': 32, 'hidden_dim': 400}                                                                                                                                                                                                                            
# neural-population-layer-1  | Best trial final validation loss: 1.915709137916565                                                                                                                                                                                                                                                              
# neural-population-layer-1  | Best trial final validation accuracy: 0.3078                                                                                                                                                                                                                                                                     
# neural-population-layer-1  | Best trial final validation fsa_inf: 0.5293553330646917                                                                                                                                                                                                                                                          
# neural-population-layer-1  | 
# neural-population-layer-1  | Test 
# neural-population-layer-1  |   Accuracy: 37.4%, Avg loss: 1.984241 

# 2025-11-03 21:19:31,165    WARNING services.py:2155 -- WARNING: The object store is using /tmp instead of /dev/shm because /dev/shm has only 67108864 bytes available. This will harm performance! You may be able to free up space by deleting files in /dev/shm. If you are inside a Docker container, you can increase /dev/shm size by passing '--shm-size=9.92gb' to 'docker run' (or add it to the run_options list in a Ray cluster config). Make sure to set this to more than 30% of available RAM.
# neural-population-layer-1  | 2025-11-03 21:19:31,307    INFO worker.py:2012 -- Started a local Ray instance.
# neural-population-layer-1  | /usr/local/lib/python3.12/site-packages/ray/_private/worker.py:2051: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# neural-population-layer-1  |   warnings.warn(
# neural-population-layer-1  | 2025-11-03 21:19:32,428    INFO tune.py:253 -- Initializing Ray automatically. For cluster usage or custom Ray initializ


date_time = datetime.now()

identifier = "mnist_v2_0"
input_dim = 784
output_dim = 10

training_noise=1.0
training_noise_probability=0.5

#stack = "linear"
stack = "population"

def clamp_transform(tensor):
    return torch.clamp(tensor, 0.0, 1.0)

def add_noise(tensor):
    if training_noise_probability <= 0.0:
        return tensor
    
    if training_noise_probability < torch.rand(1).item() or training_noise_probability >= 1.0:
        noise = torch.randn_like(tensor) * training_noise + 0.0
        return tensor + noise
    
    return tensor

def load_data(): 
    ssl._create_default_https_context = ssl._create_unverified_context 

    transform_with_noise = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(add_noise),
            transforms.Lambda(clamp_transform)
        ])

    training_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform_with_noise
    )

    test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data

def get_stack(config):
    if stack == "population":
        return nn.Sequential(
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

    return nn.Sequential(
        nn.Linear(input_dim, config["hidden_dim"]),
        nn.ReLU(), 
        nn.Linear(config["hidden_dim"], output_dim))

def run(config, data_dir=None):
        stack = get_stack(config)

        optimizer = torch.optim.AdamW(
            stack.parameters(), 
            lr=config["lr"],
            weight_decay=config["weight_decay"])
        
        loss_fn = nn.CrossEntropyLoss()

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
    model = get_stack(config)
    
    model.load_state_dict(checkpoint_data["net_state_dict"])

    model.eval()

    training_data, test_data = load_data()
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=int(config["batch_size"]), shuffle=False
    )

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    roby_scores: dict[str, float] = {}

    flatten = nn.Flatten()

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in test_loader:
            x = flatten(x)
            pred = model(x)

            base_loss = loss_fn(pred, y).item()
            test_loss += base_loss

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            roby_metric(x, pred, p=float('inf'), metric=['fsa'],
                append_to=roby_scores)
            
            roby_metric(x, pred, p=2, metric=['fsa'],
                append_to=roby_scores)
    
    test_loss /= num_batches
    correct /= size

    config["test_accuracy"] = 100*correct
    config["test_loss"] = test_loss

    copy = config.copy()
    if copy.get("stimulus") is not None:
        copy["stimulus"] = copy["stimulus"].name

    roby_scores = OrderedDict(sorted(roby_scores.items()))
    for param in roby_scores:
        mean = sum(roby_scores[param]) / len(roby_scores[param])
        std = torch.std(torch.tensor(roby_scores[param]))

        copy[f"test_{param}_mean"] = mean
        copy[f"test_{param}_std"] = std.item()

    print(f"Best trial config: {json.dumps(copy, indent=2)}")

    print("\n")

def get_config():
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([64, 128, 256]),
        "hidden_dim": tune.choice([200, 300, 400]),
    }
    
    if stack == "population":
        config["sigma"] = tune.choice([0.6, 0.8, 1.2])
        config["neurons"] = tune.choice([12])
        config["orientation"] = tune.choice([(-2,2),(-4,4)])
        config["stimulus"] = tune.choice([PreferredStimulus.RAND_NORMAL])
    
    return config

def test_best_trial(result, metric, mode):
    best_trial = result.get_best_trial(metric, mode, "last")
    best_trial.config["stack"] = stack
    best_trial.config["noise"] = training_noise
    best_trial.config["noise_probability"] = training_noise_probability
    best_trial.config["metric"] = metric
    best_trial.config["loss"] = best_trial.last_result['loss'].item()
    best_trial.config["accuracy"] = best_trial.last_result['accuracy']
    best_trial.config["fsa_inf"] = best_trial.last_result['fsa_inf']

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric=metric, mode=mode)
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        test(best_trial.config, best_checkpoint_data)   

def main(data_dir):
    config = get_config()

    load_data()

    scheduler = ASHAScheduler(
        metric="fsa_inf",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    ) 

    result: Any | None

    try:
        result = tune.run(
            partial(run, data_dir=data_dir),
            config=config,
            num_samples=50,
            scheduler=scheduler,
            max_concurrent_trials=6,
        )
    except:
        print("error")

    test_best_trial(result, metric="loss", mode="min") 

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(identifier)