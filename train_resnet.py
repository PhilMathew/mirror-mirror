from tqdm import tqdm
import torchvision
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from typing import *


def init_ds(ds_type: str):
    transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    match ds_type:
        case 'MNIST':
            train_ds, test_ds = MNIST('./data', transform=transform), MNIST('./data', train=False, transform=transform)
        case 'CIFAR10':
            train_ds, test_ds = CIFAR10('./data', transform=transform), CIFAR10('./data', train=False, transform=transform)
        case 'CIFAR100':
            train_ds, test_ds = CIFAR100('./data', transform=transform), CIFAR100('./data', train=False, transform=transform)
        case _:
            raise ValueError(f'{ds_type} is an invalid dataset type') 
    
    return train_ds, test_ds
        