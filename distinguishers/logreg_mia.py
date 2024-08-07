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


# From https://github.com/if-loops/selective-synaptic-dampening.git
def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(dataset, model, batch_size, device):
    model = model.to(device)
    model = model.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).cpu().data)
    return torch.cat(prob)


# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(retain_ds, forget_ds, test_ds, model, batch_size, device):
    retain_prob = collect_prob(retain_ds, model, batch_size, device)
    forget_prob = collect_prob(forget_ds, model, batch_size, device)
    test_prob = collect_prob(test_ds, model, batch_size, device)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


# https://arxiv.org/abs/2205.08096
def get_membership_attack_prob(retain_ds, forget_ds, test_ds, model, batch_size, device):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_ds, forget_ds, test_ds, model, batch_size, device
    )
    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()


# Rest is own work
def run_logreg_mia(
    model: nn.Module,
    model_forget_ds: Dataset,
    model_retain_ds: Dataset,
    model_test_ds: Dataset,
    batch_size: int,
    device: torch.device
) -> float:
    """
    Runs a membership inference attack using a logistic regression model

    :param model: Model to infer membership on
    :type model: nn.Module
    :param model_forget_ds: Forget set
    :type model_forget_ds: Dataset
    :param model_retain_ds: Retain set (used to train the attack)
    :type model_retain_ds: Dataset
    :param model_test_ds: Test set (used to train the attack)
    :type model_test_ds: Dataset
    :param batch_size: Number of samples in each batch
    :type batch_size: int
    :param device: Device to run the model on
    :type device: torch.device
    :return: Membership inference score for the forget set
    :rtype: float
    """
    mia_score = get_membership_attack_prob(
        retain_ds=model_retain_ds,
        forget_ds=model_forget_ds,
        test_ds=model_test_ds,
        model=model,
        batch_size=batch_size,
        device=device
    )
    
    return mia_score
