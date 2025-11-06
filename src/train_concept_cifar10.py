from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import itertools
import random
import torch

from training.training import HyperParameter, LinearNetworkTraining
from datasets.cifar10 import CIFAR10

date_time = datetime.now()

identifier = "cifar10_linear_v1_0"

seed = random.randint(1000000, 9999999)
torch.manual_seed(seed)  

for noise in [1.0]:
    population_training = LinearNetworkTraining(
        hyper_parameter=HyperParameter(
            dataset=CIFAR10(),
            hidden_dim=512,
            training_noise=noise,
            test_noise=0.2,
            batch_size=256,
            learning_rate=0.00028284242675791167,
            weight_decay=0.00015650722504916676,
            epochs=100,
            created_on=date_time,
            seed=seed,
            subset=None,
            index=0,
            identifier=identifier))

    population_training.run()



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