from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from typing import *
from torch.utils.data import Dataset


# Improves model performance (https://github.com/kuangliu/pytorch-cifar/blob/master/main.py)
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


class MembershipInferenceDataset(Dataset):
    def __init__(
        self, 
        member_dataset: Dataset, 
        non_member_dataset: Dataset
    ) -> None:
        """
        Constructs a membership inference dataset from existing member and non-member datasets.

        :param member_dataset: Dataset of members.
        :type member_dataset: Dataset
        :param non_member_dataset: Dataset of non-members.
        :type non_member_dataset: Dataset
        """
        super(MembershipInferenceDataset, self).__init__()
        self.member_ds, self.non_member_ds = member_dataset, non_member_dataset
    
    def __getitem__(self, index: int) -> Any:
        if index < len(self.member_ds):
            return *self.member_ds[index], 1
        else:
            return *self.non_member_ds[index - len(self.member_ds)], 0
    
    def __len__(self) -> int:
        return len(self.member_ds) + len(self.non_member_ds)


def init_full_ds(ds_type: str, use_random_train_aug: bool = False) -> Tuple[Dataset, Dataset, int, int]:
    """
    Initialize the full dataset and figure out the number of classes and input channels.

    :param ds_type: Name of the dataset to use. See README.md for more details on available datasets.
    :type ds_type: str
    :param use_random_train_aug: Whether to use random training augmentations, defaults to False
    :type use_random_train_aug: bool, optional
    :raises ValueError: If ds_type is not one of the available datasets.
    :return: The full training dataset, the test dataset, the number of classes, and the number of input channels
    :rtype: Tuple[Dataset, Dataset, int, int]
    """
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
            if use_random_train_aug:
                train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.Resize(64),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                    ]
                )
            else:
                train_transform = transforms.Compose(
                    [
                        transforms.Resize(64),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                    ]
                )
            
            test_transform = transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ]
            )
            
            train_ds = CIFAR10('./data', transform=train_transform, download=True)
            test_ds = CIFAR10('./data', train=False, transform=test_transform, download=True)
            num_classes, in_channels = 10, 3
        case 'CIFAR100':
            if use_random_train_aug:
                train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                    ]
                )
            else:
                train_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                    ]
                )
            
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ]
            )
            
            train_ds = CIFAR100('./data', transform=train_transform, download=True)
            test_ds = CIFAR100('./data', train=False, transform=test_transform, download=True)
            num_classes, in_channels = 100, 3
        case _:
            raise ValueError(f'{ds_type} is an invalid dataset type') 
    
    return train_ds, test_ds, num_classes, in_channels
