from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from typing import *
from torch.utils.data import Dataset


def init_full_ds(ds_type: str) -> Tuple[Dataset, Dataset]:
    
    match ds_type:
        case 'MNIST':
            transform = transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.Normalize(0.5, 0.5)
                ]
            )
            train_ds, test_ds = MNIST('./data', transform=transform, download=True), MNIST('./data', train=False, transform=transform, download=True)
        case 'CIFAR10':
            transform = transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
            train_ds, test_ds = CIFAR10('./data', transform=transform, download=True), CIFAR10('./data', train=False, transform=transform, download=True)
        case 'CIFAR100':
            transform = transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
            train_ds, test_ds = CIFAR100('./data', transform=transform, download=True), CIFAR100('./data', train=False, transform=transform, download=True)
        case _:
            raise ValueError(f'{ds_type} is an invalid dataset type') 
    
    return train_ds, test_ds
