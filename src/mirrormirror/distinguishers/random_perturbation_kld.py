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

from ..components.datasets import CIFAR_MEAN, CIFAR_STD


class RandomPerturbationDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        perturbation_fn: Callable[[torch.Tensor], torch.Tensor],
        seed: int = 0
    ):
        super(RandomPerturbationDataset, self).__init__()
        self.dataset = dataset
        self.perturbation_fn = perturbation_fn
        self.seed = seed
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        torch.manual_seed(self.seed)
        orig_x, y = self.dataset[index]
        x = self.perturbation_fn(orig_x)
        
        return x, orig_x, y


def compute_kld_over_perturbations(
    candidate_model: nn.Module, 
    original_model: nn.Module, 
    dataset: Dataset, 
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
    
    perturbation_ds = RandomPerturbationDataset(dataset, perturbation_fn, seed)
    perturbation_dl = DataLoader(perturbation_ds, batch_size=batch_size, num_workers=num_workers)
    
    with torch.no_grad():
        query_preds, orig_preds = [], []
        for batch in perturbation_dl:
            x, orig_x, _ = batch
            x, orig_x = x.to(device), orig_x.to(device)
            query_pred = F.log_softmax(candidate_model(x), dim=-1).cpu()
            orig_pred = F.log_softmax(original_model(x), dim=-1).cpu()
            query_preds.append(query_pred)
            orig_preds.append(orig_pred)
        query_preds, orig_preds = torch.cat(query_preds, dim=0), torch.cat(orig_preds, dim=0)
        kld = F.kl_div(query_preds, orig_preds, reduction='batchmean', log_target=True).item()
    
    return kld


def compute_svm_consistency(
    candidate_model: nn.Module, 
    original_model: nn.Module, 
    dataset: Dataset, 
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
    
    with torch.no_grad():
        scores = []
        for orig_x, _ in dataset:
            # TODO: Clean this up
            orig_x = orig_x.to(device)
            preds = []
            for _ in range(10):
                x = perturbation_fn(orig_x.cpu()).to(device)
                pred = F.log_softmax(
                    candidate_model(x.reshape(1, *x.shape)),
                    dim=-1
                ).cpu()
                preds.append(pred)
            clean_pred = F.log_softmax(
                candidate_model(orig_x.reshape(1, *orig_x.shape)),
                dim=-1
            ).cpu()
            preds.append(clean_pred)
            preds = torch.cat(preds, dim=0).numpy()
            
            
            # Compute score
            clf = OneClassSVM()
            clf.fit(preds)
            clf_scores = clf.score_samples(preds)
            curr_score = np.mean((clf_scores[:-1] - clf_scores[-1])**2)
            
            scores.append(curr_score)
    
    return np.mean(scores)


def compute_kld_over_pixle(
    candidate_model: nn.Module, 
    original_model: nn.Module, 
    dataset: Dataset,
    seed: int, 
    device: torch.device, 
    batch_size: int = 128, 
    num_workers: int = 16
) -> float:
    candidate_model = candidate_model.to(device)
    original_model = original_model.to(device)
    candidate_model.eval()
    original_model.eval()
    
    # Run Pixle
    atk = torchattacks.Pixle(candidate_model)
    atk.set_normalization_used(CIFAR_MEAN, CIFAR_STD)
    atk.set_mode_targeted_by_function(lambda x, y: y)

    clean_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    query_preds, orig_preds = [], []
    with torch.no_grad():
        for imgs, labels in clean_dl:
            batch_adv_imgs = atk(imgs, labels)
            # preds = F.softmax(candidate_model(batch_adv_imgs.float().cuda()), dim=-1).argmax(dim=-1)
            # adv_imgs.extend(batch_adv_imgs)    
            # x, orig_x, _ = batch
            # x, orig_x = x.to(device), orig_x.to(device)
            query_pred = F.log_softmax(candidate_model(batch_adv_imgs), dim=-1).cpu()
            orig_pred = F.log_softmax(original_model(batch_adv_imgs), dim=-1).cpu()
            query_preds.append(query_pred)
            orig_preds.append(orig_pred)
        query_preds, orig_preds = torch.cat(query_preds, dim=0), torch.cat(orig_preds, dim=0)
        kld = F.kl_div(query_preds, orig_preds, reduction='batchmean', log_target=True).item()
    
    return kld
