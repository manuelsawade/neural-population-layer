from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import itertools
import json
from pathlib import Path
import random
import torch

from datasets.mnist import MNIST
from training.training import HyperParameter, LinearNetworkTraining
from datasets.cifar10 import CIFAR10

date_time = datetime.now()

identifier = "cifar10_evaluation_linear"
path = f"./experiments/{identifier}/tuning/"

folder_path = Path(path)
file_paths = sorted(folder_path.glob("linear*.json"))

def run(p: tuple):
    print("load file:", p)
    try:
        with open(p, "r") as f:
            tuning_result = json.loads(f.read())

            seed = random.randint(1000000, 9999999)
            torch.manual_seed(seed)  

            for index in [range(1)]:
                population_training = LinearNetworkTraining(
                    hyper_parameter=HyperParameter(
                        dataset=CIFAR10(),
                        hidden_dim=tuning_result["hidden_dim"],
                        training_noise=tuning_result["noise"],
                        test_noise=0.2,
                        batch_size=tuning_result["batch_size"],
                        learning_rate=tuning_result["lr"],
                        weight_decay=tuning_result["weight_decay"],
                        epochs=100,
                        created_on=date_time,
                        seed=seed,
                        subset=None,
                        index=index,
                        identifier=identifier,
                        stack=tuning_result["stack"]))

                population_training.run()

    except Exception as e:
        print(f"could not read: {e}")

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(run, file_paths))

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(run, file_paths))



# torch.Size([128])
# torch.Size([32, 128])
# torch.Size([32, 128])

# mask   torch.Size([128])
# diff   torch.Size([32, 128])
# num    torch.Size([128])
# denum  torch.Size([32])
# denum  torch.Size([32, 1])
# a      torch.Size([32, 128])
# a_norm torch.Size([32, 128])