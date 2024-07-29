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


class RandomImageDataset(Dataset):
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
        x, y = self.dataset[index]
        x = self.perturbation_fn(x)
        
        return x, y


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
        for batch in tqdm(perturbation_dl, desc='Getting Model Outputs'):
            x, _ = batch
            x = x.to(device)
            query_pred = F.log_softmax(candidate_model(x), dim=-1).cpu()
            orig_pred = F.log_softmax(original_model(x), dim=-1).cpu()
            query_preds.append(query_pred)
            orig_preds.append(orig_pred)
        query_preds, orig_preds = torch.cat(query_preds, dim=0), torch.cat(orig_preds, dim=0)
        kld = F.kl_div(query_pred, orig_pred, reduction='batchmean', log_target=True).item()
    
    return kld
