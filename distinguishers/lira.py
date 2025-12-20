import random
from typing import *

import pandas as pd
import scipy.stats
import torch
import scipy
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset, Dataset

from components.models import build_resnet50
from utils.train_utils import train_model


class MIDataset(Dataset):
    def __init__(
        self, 
        member_dataset: Dataset, 
        non_member_dataset: Dataset
    ) -> None:
        super(MIDataset, self).__init__()
        self.member_ds, self.non_member_ds = member_dataset, non_member_dataset
    
    def __getitem__(self, index: int) -> Any:
        if index < len(self.member_ds):
            return *self.member_ds[index], 1
        else:
            return *self.member_ds[index], 0
    
    def __len__(self) -> int:
        return len(self.member_ds) + len(self.non_member_ds)


def phi(p: float) -> float:
    return np.log(p / (1 - p))


def get_lira_results(
    member_ds: Dataset,
    non_member_ds: Dataset,
    data_distro_ds: Dataset, 
    model: nn.Module, 
    num_shadow_models: int, 
    num_classes: int, 
    in_channels: int, 
    device: torch.device, 
    batch_size: int, 
    num_epochs: int
) -> pd.DataFrame:
    # Using offline variant of LiRA from Carlini's "Membership Inference Attacks from First Principles"
    # TODO: Ensure models aren't being arbitrarily thrown everywhere to kill the GPU 
    mem_ds = MIDataset(member_ds, non_member_ds)

    all_scores = {k: [] for k in ('ds_ind', 'mi_score', 'member_label')}
    for i, (x, y, member_label) in enumerate(mem_ds):
        conf_out = []
        for j in range(num_shadow_models):
            # Create "out" dataset
            subset_inds = random.sample(range(len(data_distro_ds)), k=int(len(data_distro_ds) * 0.5)) # randomly choose 50% of the data
            out_inds = [k for k in subset_inds if k != i]
            D_out = Subset(data_distro_ds, out_inds)
            
            # Create and train "out" shadow model
            M_out = build_resnet50(num_classes, in_channels)
            train_model(M_out, D_out, device, batch_size=batch_size, num_epochs=num_epochs)
            
            # Get scores from shadow models
            score_out = F.softmax(M_out(x.to(device)), dim=-1)[y].item()
            conf_out.append(phi(score_out))
            
        # Get estimates of mean and variance, then construct gaussians
        mu_out = np.mean(conf_out)
        var_out = np.var(conf_out)
        dist_out = scipy.stats.norm(mu_out, var_out)
        
        # Get membership inference score
        model = model.to(device)
        conf_obs = F.softmax(model(x.to(device)), dim=-1)[y].item()
        mi_score = dist_out.cdf(conf_obs)
        
        # Track the numbers
        all_scores['ds_ind'].append(i)
        all_scores['mi_score'].append(mi_score)
        all_scores['member_label'].append(member_label)
    all_scores_df = pd.DataFrame(all_scores)
    
    return all_scores_df
