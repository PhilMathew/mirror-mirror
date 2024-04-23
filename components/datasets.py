from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from typing import *
from torch.utils.data import Dataset


class MembershipInferenceDataset(Dataset):
    def __init__(
        self, 
        member_dataset: Dataset, 
        non_member_dataset: Dataset
    ) -> None:
        super(MembershipInferenceDataset, self).__init__()
        self.member_ds, self.non_member_ds = member_dataset, non_member_dataset
    
    def __getitem__(self, index: int) -> Any:
        if index < len(self.member_ds):
            return *self.member_ds[index], 1
        else:
            return *self.non_member_ds[index - len(self.member_ds)], 0
    
    def __len__(self) -> int:
        return len(self.member_ds) + len(self.non_member_ds)


def init_full_ds(ds_type: str) -> Tuple[Dataset, Dataset, int, int]:
    match ds_type:
        case 'MNIST':
            transform = transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.Normalize(0.5, 0.5)
                ]
            )
            
            train_ds = MNIST('./data', transform=transform, download=True) 
            test_ds = MNIST('./data', train=False, transform=transform, download=True)
            num_classes, in_channels = 10, 1
        case 'CIFAR10':
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.RandomResizedCrop(size=(64, 64), antialias=True),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            
            train_ds = CIFAR10('./data', transform=train_transform, download=True)
            test_ds = CIFAR10('./data', train=False, transform=test_transform, download=True)
            num_classes, in_channels = 10, 3
        case 'CIFAR100':
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.RandomResizedCrop(size=(64, 64), antialias=True),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
            
            train_ds = CIFAR100('./data', transform=train_transform, download=True)
            test_ds = CIFAR100('./data', train=False, transform=test_transform, download=True)
            num_classes, in_channels = 100, 3
        case _:
            raise ValueError(f'{ds_type} is an invalid dataset type') 
    
    return train_ds, test_ds, num_classes, in_channels
