from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import itertools
import random
import torch

from training.training import HyperParameter, LinearNetworkTraining
from datasets.mnist import MNIST

date_time = datetime.now()

identifier = "mnist_stateful_preferred_v1_0"

parameter = {
    "dataset": [MNIST()],
    "runs": [10],
    "epochs": [100],
    "noise": [0.0, 0.5, 1.0],
    "hidden_dim": [200],
    "batch_size": [32],
    "learning_rate": [0.00001],
    "weight_decay": [0.0001],
}

all_parameter = list(itertools.product(*parameter.values()))
print(f"max runs: {len(all_parameter) * parameter["runs"][0]}")

def run(params: tuple):
    dataset, runs, epochs, noise, hidden_dim, batch_size, learning_rate, weight_decay = params
    for i in range(runs):
        seed = random.randint(1000000, 9999999)
        torch.manual_seed(seed)  
        
        population_training = LinearNetworkTraining(
            hyper_parameter=HyperParameter(
                dataset=dataset,
                hidden_dim=hidden_dim,
                training_noise=noise,
                test_noise=0.2,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs,
                created_on=date_time,
                seed=seed,
                subset=None,
                index=i,
                identifier=identifier))

        population_training.run()

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(run, all_parameter))

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