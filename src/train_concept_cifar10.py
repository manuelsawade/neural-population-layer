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

population_training = LinearNetworkTraining(
    hyper_parameter=HyperParameter(
        dataset=CIFAR10(),
        hidden_dim=200,
        training_noise=0.0,
        test_noise=0.2,
        batch_size=64,
        learning_rate=0.0005609176336288977,
        weight_decay=0.01,
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