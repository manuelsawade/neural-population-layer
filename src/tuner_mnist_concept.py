#!/usr/bin/python3

from collections import OrderedDict
from datetime import datetime
from functools import partial
import json
from typing import Any
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from torch.utils.data import random_split
from ray import tune
from ray.tune import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
from ray.tune.schedulers import ASHAScheduler

from metrics.roby import roby_metric
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import ssl
import torchvision

from activations.neuron import NeuronPopulation, PreferredStimulus
from decoder import WeightedAverageDecoder
from populations import Distribution, SineWave, TuningCurve
from activations.sine_layer import PreferredValueActivation, PreferredValueInitializer
from tuner_data_transforms import add_noise, clamp_transform

date_time = datetime.now()

identifier = "mnist_evaluation_preferred_value"
input_dim = 784
output_dim = 10

#stack = "linear"
stack = "preferred_value"
target_metric = "fsa_inf_std"
target_mode="min"

training_noise=1.0

scheduler = ASHAScheduler(
    metric=target_metric,
    mode=target_mode,
    max_t=50,        
    grace_period=4,
    reduction_factor=2
) 

def get_config():
    config = {
        "lr": tune.loguniform(1e-9, 1e-6),
        "weight_decay": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([4, 8, 16]),
        "hidden_dim": tune.choice([128, 256]),
    }
    
    if stack == "preferred_value":
        config["freq"] = tune.choice([4, 6, 8, 10, 12, 14, 16, 18])
        config["phase"] = tune.loguniform(0.5, 1.5)
        config["amp"] = tune.choice([0.5, 1.0, 1.5])
        config["distribution"] = tune.choice([Distribution.ZERO_MEAN, Distribution.ZERO_BASE])
        config["initializer"] = tune.choice([
            PreferredValueInitializer.SINE_WAVE, 
            PreferredValueInitializer.RANDOM_NORMAL, 
            PreferredValueInitializer.RANDOM_UNIFORM])
        config["requires_grad"] = tune.choice([True, False])
    
    return config

def load_data(): 
    ssl._create_default_https_context = ssl._create_unverified_context 

    transform_with_noise = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(partial(add_noise, training_noise=training_noise)),
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
    if stack == "preferred_value":
        match config["initializer"]:
            case PreferredValueInitializer.SINE_WAVE:
                preference = SineWave(
                    config["hidden_dim"], 
                    freq=config["freq"],
                    phase=config["phase"],
                    amp=config["amp"],
                    dist=config["distribution"],
                    requires_grad=config["requires_grad"]
                )
            case PreferredValueInitializer.RANDOM_NORMAL:
                preference = torch.randn(config["hidden_dim"])
            case PreferredValueInitializer.RANDOM_UNIFORM:
                preference = torch.rand(config["hidden_dim"])

        return nn.Sequential(
            nn.Linear(input_dim, config["hidden_dim"]),
            PreferredValueActivation(initialized=preference),
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
            train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=2
        )
        valloader = torch.utils.data.DataLoader(
            val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=2
        )

        flatten = nn.Flatten()

        for epoch in range(start_epoch, 10): 
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()

                flatten = nn.Flatten()

                inputs = flatten(inputs)

                outputs = stack(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

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

                mean = sum(val_fsa_scores["fsa_inf"]) / len(val_fsa_scores["fsa_inf"])
                std = torch.std(torch.tensor(val_fsa_scores["fsa_inf"])).item()

                tune.report(
                    {
                        "loss": val_loss / val_steps, 
                        "accuracy": correct / total, 
                        "fsa_inf_mean": mean,
                        "fsa_inf_std": std,
                    },
                    checkpoint=checkpoint,
                )

        print("Finished Training") 

def test(config, checkpoint_data, scope):
    model = get_stack(config)
    
    model.load_state_dict(checkpoint_data["net_state_dict"])

    model.eval()

    _, test_data = load_data()
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
    if copy.get("distribution") is not None:
        copy["distribution"] = copy["distribution"].name

    if copy["initializer"] != PreferredValueInitializer.SINE_WAVE:
        for remove in ["freq","phase","amp","distribution","requires_grad"]:
            del copy[remove]

    roby_scores = OrderedDict(sorted(roby_scores.items()))
    for param in roby_scores:
        mean = sum(roby_scores[param]) / len(roby_scores[param])
        std = torch.std(torch.tensor(roby_scores[param]))

        copy[f"test_{param}_mean"] = mean
        copy[f"test_{param}_std"] = std.item()

    result = json.dumps(copy, indent=2)

    print(f"Best trial config: {result}")

    date = date_time.strftime("%Y_%m_%d_%H_%M_%S")

    with open(f"./experiments/tuning/{stack}_{scope}_{identifier}_{date}.json", mode='w', newline='') as file:
        file.write(result)

    print("\n")

def test_best_trial(result, metric, mode, scope):
    best_trial = result.get_best_trial(metric, mode, scope)
    best_trial.config["stack"] = stack
    best_trial.config["noise"] = training_noise
    best_trial.config["metric"] = metric
    best_trial.config["target_metric"] = target_metric
    best_trial.config["loss"] = best_trial.last_result['loss'].item()
    best_trial.config["accuracy"] = best_trial.last_result['accuracy']
    best_trial.config["fsa_inf_mean"] = best_trial.last_result['fsa_inf_mean']
    best_trial.config["fsa_inf_std"] = best_trial.last_result['fsa_inf_std']

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric=metric, mode=mode)
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        test(best_trial.config, best_checkpoint_data, scope)   

def main(data_dir):
    config = get_config()

    load_data()

    result: Any | None

    try:
        result = tune.run(
            partial(run, data_dir=data_dir),
            config=config,
            num_samples=30,
            scheduler=scheduler,
            max_concurrent_trials=12,
        )
    except:
        print("error")

    test_best_trial(result, metric="loss", mode="min", scope="last") 
    test_best_trial(result, metric="loss", mode="min", scope="last-5-avg") 
    test_best_trial(result, metric="loss", mode="min", scope="last-10-avg")

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(identifier)