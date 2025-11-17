#!/usr/bin/python3

from collections import OrderedDict
from datetime import datetime
from functools import partial
import json
import os
import shutil
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
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search import ConcurrencyLimiter
from ray.exceptions import RayError
from ray.tune.search.optuna import OptunaSearch

from metrics.roby import roby_metric
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import ssl
import torchvision
import optuna

from activations.neuron import NeuronPopulation, PreferredStimulus
from decoder import CircularMeanDecoder, WeightedAverageDecoder
from populations import CircularTuningCurve, Distribution, MexicanHat, SineWave, TuningCurve
from activations.dynamic import SoftmaxGaussianActivation
from activations.sine_layer import PreferredValueActivation, PreferredValueInitializer
from tuner_data_transforms import add_noise, clamp_transform

def run_tuning(training_noise, stack):
    date_time = datetime.now()

    #training_noise=0.5

    dataset = "mnist"
    #dataset = "cifar10"
    identifier = f"{dataset}_evaluation"

    warmup=0

    #stack = "linear"
    #stack = "population"
    #stack = "population_circular"
    #stack = "population_encoding"
    #stack = "softmax_gaussian"
    #stack = "preferred_value"
    target_metric = "loss_norm" 
    target_mode="min"

    def get_config():
        config = {
            "lr": tune.loguniform(1e-9, 1e-4),
            "weight_decay": tune.loguniform(1e-6, 1e-4),
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
        
        if "population" in stack:
            config["sigma"] = tune.loguniform(0.5, 1.5)
            config["neurons"] = tune.choice([8, 12, 16])
            config["orientation"] = tune.choice([(0,1),(-1,1),(-2,2),(-4,4),(-5,5)])
            config["stimulus"] = tune.choice([PreferredStimulus.RAND_NORMAL, PreferredStimulus.LINEAR, PreferredStimulus.RAND_UNIFORM])
            config["output_encoding"] = True if stack == "population_encoding" else False

        if stack == "softmax_gaussian":
            config["activation"] = tune.choice([TuningCurve, MexicanHat])
            config["alpha"] = tune.loguniform(1.0, 10.0)
            config["sigma"] = tune.loguniform(0.1, 0.3)
            config["normalize"] = tune.choice([True, False]) 
        
        return config

    sampler = optuna.samplers.TPESampler(
        multivariate=True,  
        group=True           
    )

    scheduler = ASHAScheduler(
        metric=target_metric,
        mode=target_mode,
        time_attr="training_iteration",
        max_t=50,        
        grace_period=2,
        reduction_factor=2
    ) 

    search_alg = OptunaSearch(
        sampler=sampler,
        space=get_config(),
        metric=["fsa_inf_mean_norm", "fsa_inf_mean_diff"],
        mode=["max", "min"]
    )

    def load_data(): 
        ssl._create_default_https_context = ssl._create_unverified_context 

        transform_with_noise = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(partial(add_noise, training_noise=training_noise)),
                transforms.Lambda(clamp_transform)
            ])
        
        dataset_loader = torchvision.datasets.MNIST if dataset == "mnist" else torchvision.datasets.CIFAR10

        training_data = dataset_loader(
            root="data",
            train=True,
            download=True,
            transform=transform_with_noise
        )

        test_data = dataset_loader(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        return training_data, test_data

    input_dim = 784 if dataset == "mnist" else 32 * 32 * 3
    output_dim = 10

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
                nn.LazyLinear(output_dim))
        
        if stack == "population_encoding":
            return nn.Sequential(
                nn.Linear(input_dim, config["hidden_dim"]),
                NeuronPopulation(
                    config["hidden_dim"], 
                    sigma=config["sigma"],#sigma=hyper_parameter.sigma,
                    neurons=config["neurons"],#neurons=hyper_parameter.neurons,
                    orientation=config["orientation"],#orientation=hyper_parameter.orientation,
                    activation=TuningCurve(readout=WeightedAverageDecoder()),
                    stimulus=config["stimulus"],
                    encoded_output=True),
                nn.LazyLinear(output_dim))
        
        if stack == "population_circular":
            return nn.Sequential(
                nn.Linear(input_dim, config["hidden_dim"]),
                NeuronPopulation(
                    config["hidden_dim"], 
                    sigma=config["sigma"],#sigma=hyper_parameter.sigma,
                    neurons=config["neurons"],#neurons=hyper_parameter.neurons,
                    orientation=config["orientation"],#orientation=hyper_parameter.orientation,
                    activation=CircularTuningCurve(readout=CircularMeanDecoder()),
                    stimulus=config["stimulus"]),
                nn.LazyLinear(output_dim))
        
        if stack == "softmax_gaussian":
            return nn.Sequential(
                nn.Linear(input_dim, config["hidden_dim"]),
                SoftmaxGaussianActivation(
                    activation=config["activation"](readout=None), 
                    alpha=config["alpha"], 
                    sigma=config["sigma"], 
                    normalize=config["normalize"]), 
                nn.Linear(config["hidden_dim"], output_dim))

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

                    val_fsa_scores = checkpoint_state["val_fsa_scores"]
                    loss_diff = checkpoint_state["loss_diff"]
                    fsa_max = checkpoint_state["fsa_max"]
            else:
                start_epoch = 0
                val_fsa_scores = {}
                loss_diff = 0
                fsa_max = 0

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

            for epoch in range(start_epoch, 10 + warmup): 
                train_fsa_scores = {}
                
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()

                    flatten = nn.Flatten()

                    inputs = flatten(inputs)

                    outputs = stack(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    roby_metric(outputs, inputs, p=float('inf'), metric=['fsa'],
                        append_to=train_fsa_scores)

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

                if epoch < warmup:
                    continue

                mean = sum(val_fsa_scores["fsa_inf"]) / len(val_fsa_scores["fsa_inf"])
                std = torch.std(torch.tensor(val_fsa_scores["fsa_inf"])).item()

                loss = val_loss / val_steps
                
                if epoch == 0:
                    loss_diff = loss
                    fsa_max = mean

                mean_norm = mean - fsa_max
                loss_norm = loss - loss_diff

                

                # if len(fsa_diff) < 1:
                #     fsa_diff.append(mean)
            
                # fsa_diff.append(1 - (max(fsa_diff) / fsa_diff[0]))

                checkpoint_data = {
                    "epoch": epoch,
                    "net_state_dict": stack.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_fsa_scores": val_fsa_scores,
                    "loss_diff": loss_diff,
                    "fsa_max": fsa_max
                }
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)

                    mean_train = sum(train_fsa_scores["fsa_inf"][:10]) / len(train_fsa_scores["fsa_inf"][:10])

                    tune.report(
                        {
                            "training_iteration": epoch + 1,
                            "loss": val_loss / val_steps, 
                            "accuracy": correct / total, 
                            "fsa_inf_mean": mean,
                            "fsa_inf_mean_norm": mean_norm,
                            "fsa_inf_std": std,
                            "fsa_inf_mean_diff": mean_train - mean,
                            "loss_norm": loss_norm
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
        if copy.get("stimulus") is not None:
            copy["stimulus"] = copy["stimulus"].name

        if stack == "softmax_gaussian":
            copy["activation"] = copy["activation"].name

        if copy.get("distribution") is not None:
            copy["distribution"] = copy["distribution"].name

        if copy.get("initializer") is not None and copy["initializer"] != PreferredValueInitializer.SINE_WAVE:
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

        with open(f"./experiments/tuning/{stack}/{stack}_{identifier}_{copy["metric"]}_{scope}_{date}.json", mode='w', newline='') as file:
            file.write(result)

        print("\n")

    def test_best_trial(result, metric, mode, scope="last"):
        best_trial = result.get_best_trial(metric, mode, scope)
        best_trial.config["stack"] = stack
        best_trial.config["noise"] = training_noise
        best_trial.config["metric"] = metric
        best_trial.config["target_metric"] = target_metric
        best_trial.config["loss"] = best_trial.last_result['loss'].item()
        best_trial.config["accuracy"] = best_trial.last_result['accuracy']
        best_trial.config["fsa_inf_mean"] = best_trial.last_result['fsa_inf_mean']
        best_trial.config["fsa_inf_std"] = best_trial.last_result['fsa_inf_std']
        best_trial.config["fsa_inf_mean_diff"] = best_trial.last_result['fsa_inf_mean_diff']

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
                #config=config,
                num_samples=30,
                scheduler=scheduler,
                search_alg=search_alg,
                max_concurrent_trials=12
            )
        except RayError as e:
            print(f"error: {e}")

        test_best_trial(result, metric=target_metric, mode="min")

        folder = Path('/Users/manuelsawade/ray_results')
        keep_count = 0
        for dir in sorted(folder.glob("run*"), reverse=True):
            if keep_count < 5:
                keep_count = keep_count + 1
                continue
            try:
                shutil.rmtree(dir)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (dir, e))
    
    main(identifier)

if __name__ == "__main__":
    for stack in ["preferred_value"]:
        for noise in [0.5]:
            run_tuning(noise, stack)