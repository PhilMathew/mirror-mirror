from torchvision.datasets import MNIST, CIFAR10, CIFAR100, Imagenette
from torchvision import transforms
from typing import *
from torch.utils.data import Dataset
from BackdoorBox.core.attacks.BadNets import CreatePoisonedDataset as poison_with_badnets
from BackdoorBox.core.attacks.WaNet import CreatePoisonedDataset as poison_with_wanet
import torch


CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) # Improves model performance (https://github.com/kuangliu/pytorch-cifar/blob/master/main.py)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


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


def init_full_ds(ds_type: str, use_random_train_aug: bool = False, do_poison: bool = False, **kwargs) -> Tuple[Dataset, Dataset, int, int]:
    """
    Initialize the full dataset and figure out the number of classes and input channels.

    :param ds_type: Name of the dataset to use. See README.md for more details on available datasets.
    :type ds_type: str
    :param use_random_train_aug: Whether to use random training augmentations, defaults to False
    :type use_random_train_aug: bool, optional
    :param do_poison: Whether to poison the dataset, defaults to False
    :type do_poison: bool, optional
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
        case 'Imagenette':
            if use_random_train_aug:
                train_transform = transforms.Compose(
                    [   
                        # transforms.Resize(320),
                        # transforms.CenterCrop(128),
                        transforms.RandomResizedCrop(128, scale=(0.35, 1)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ]
                )
            else:
                train_transform = transforms.Compose(
                    [
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ]
                )
            
            test_transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ]
            )
            train_ds = Imagenette('./data', split='train', size='320px', transform=train_transform, download=True)
            test_ds = Imagenette('./data', split='val', size='320px', transform=test_transform, download=True)
            num_classes, in_channels = 10, 3
        case _:
            raise ValueError(f'{ds_type} is an invalid dataset type') 
    
    if do_poison:
        train_ds = poison_dataset(train_ds, target_label=0, poison_rate=kwargs.get('poison_rate', 0), poison_type=kwargs.get('poison_type', None))
        test_ds = poison_dataset(test_ds, target_label=0, poison_rate=1.0, poison_type=kwargs.get('poison_type', None))
    
    return train_ds, test_ds, num_classes, in_channels


def gen_grid(height, k): # from https://github.com/THUYimingLi/BackdoorBox/blob/main/tests/test_WaNet.py
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = torch.nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid


def poison_dataset(orig_ds: Dataset, target_label: int, poison_rate: float, poison_type: str = None) -> Dataset:
    assert type(orig_ds) in (MNIST, CIFAR10), 'Poisoning only supported for MNIST and CIFAR10'
    match poison_type:
        case 'badnets': 
            poisoned_ds = poison_with_badnets(
                benign_dataset=orig_ds,
                y_target=target_label,
                poisoned_rate=poison_rate,
                pattern=None,
                weight=None,
                poisoned_transform_index=0,
                poisoned_target_transform_index=0
            )
        case 'wanet':
            identity_grid, noise_grid = gen_grid(32, 4)
            poisoned_ds = poison_with_wanet(
                benign_dataset=orig_ds,
                y_target=target_label,
                poisoned_rate=poison_rate,
                identity_grid=identity_grid,
                noise_grid=noise_grid,
                noise=False,
                poisoned_transform_index=0,
                poisoned_target_transform_index=0
            )
        case _:
            raise ValueError(f'{poison_type} is an invalid poison type')
    
    return poisoned_ds
