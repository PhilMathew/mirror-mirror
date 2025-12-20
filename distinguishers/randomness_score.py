import random
from typing import *

import pandas as pd
import scipy.stats
import torch
import scipy
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Subset, Dataset, DataLoader
import torchattacks
from sklearn.svm import OneClassSVM

from components.datasets import CIFAR_MEAN, CIFAR_STD


def compute_model_randomness(
    candidate_model: nn.Module, 
    original_model: nn.Module, 
    dataset: Dataset,
    num_classes: int,
    perturbation_fn: Callable[[torch.Tensor], torch.Tensor], 
    seed: int, 
    device: torch.device, 
    batch_size: int = 32, 
    num_workers: int = 16
) -> float:
    candidate_model = candidate_model.to(device)
    original_model = original_model.to(device)
    candidate_model.eval()
    original_model.eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    loss_diffs = []
    for imgs, labels in dataloader:
        with torch.no_grad(): # TODO: Try comparing to orig model if this doesn't work
            imgs, labels = imgs.to(device), labels.to(device)
            preds = candidate_model(imgs)
            loss = F.cross_entropy(preds, labels).item()
            loss_diff = np.abs(loss - np.log(1 / num_classes))
            loss_diffs.append(loss_diff)
    
    return np.mean(loss_diff)
