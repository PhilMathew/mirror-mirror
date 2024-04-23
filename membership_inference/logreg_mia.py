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
from sklearn.linear_model import LogisticRegression

from components.datasets import MembershipInferenceDataset


# From https://github.com/if-loops/selective-synaptic-dampening.git
def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def train_mia_logreg(
    model: nn.Module,
    member_train_ds: Dataset,
    non_member_train_ds: Dataset,
    batch_size: int,
    device: torch.device
) -> LogisticRegression:
    
    mia_train_ds = MembershipInferenceDataset(member_train_ds, non_member_train_ds)
    mia_train_dl = DataLoader(mia_train_ds, batch_size=batch_size)
    model = model.to(device)
    model.eval()
    
    model_preds, member_labels = [], []
    with torch.no_grad():
        for batch in tqdm(mia_train_dl, desc='Getting Model Outputs'):
            x, _, member_label = batch
            x = x.to(device)
            pred = F.softmax(model(x), dim=-1).cpu()
            pred = entropy(pred, dim=-1)
            model_preds.append(pred)
            member_labels.append(member_label)
    model_preds = torch.cat(model_preds, dim=0).numpy().reshape(-1, 1)
    member_labels = torch.cat(member_labels, dim=0).numpy()
    
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", multi_class="multinomial")
    clf.fit(model_preds, member_labels)
    
    return clf


def run_logreg_mia(
    model: nn.Module,
    model_forget_ds: Dataset,
    model_retain_ds: Dataset,
    model_test_ds: Dataset,
    batch_size: int,
    device: torch.device
):
    clf = train_mia_logreg(
        model=model,
        member_train_ds=model_retain_ds,
        non_member_train_ds=model_test_ds,
        batch_size=batch_size,
        device=device
    )
    model = model.to(device)
    model.eval()
    
    mia_test_dl = DataLoader(model_forget_ds, batch_size=batch_size)
    model_preds = []
    with torch.no_grad():
        for batch in tqdm(mia_test_dl, desc='Getting Model Outputs'):
            x, _ = batch
            x = x.to(device)
            pred = F.softmax(model(x), dim=-1).cpu()
            pred = entropy(pred, dim=-1)
            model_preds.append(pred)
    model_preds = torch.cat(model_preds, dim=0).numpy().reshape(-1, 1)
    
    mia_preds = clf.predict(model_preds)
    mia_score = np.mean(mia_preds)
    
    return mia_score, int(np.sum(mia_preds == 1))
