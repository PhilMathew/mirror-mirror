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

from distinguishers.logreg_mia import collect_prob, entropy


def get_adv_advantage(
    forget_ds,
    test_ds, 
    candidate_model, 
    batch_size, 
    device
):  
    # Need to split out shadow forget and test sets to train the MIA models
    # NOTE: Currently just using half of the datasets
    shadow_forget = Subset(
        forget_ds, 
        indices=random.sample(
            list(range(len(forget_ds))), 
            k=(len(forget_ds) // 2)
        )
    )
    shadow_test = Subset(
        test_ds, 
        indices=random.sample(
            list(range(len(test_ds))), 
            k=(len(forget_ds) // 2)
        )
    )

    # Get probabilities
    shadow_forget_prob = collect_prob(shadow_forget, candidate_model, batch_size, device)
    shadow_test_prob = collect_prob(shadow_test, candidate_model, batch_size, device)
    forget_prob = collect_prob(forget_ds, candidate_model, batch_size, device)
    test_prob = collect_prob(test_ds, candidate_model, batch_size, device)
    
    # Get labels and convert to entropy values (0 is forget set, 1 is test set)
    # MIA shadow training data
    X_mia = ( 
        torch.cat([entropy(shadow_test_prob), entropy(shadow_forget_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_mia = np.concatenate([np.ones(len(shadow_test_prob)), np.zeros(len(shadow_forget_prob))])

    # Forget set
    X_forget = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_forget = np.concatenate([np.zeros(len(forget_prob))])

    # Test set
    X_test = entropy(test_prob).cpu().numpy().reshape(-1, 1)
    Y_test = np.concatenate([np.ones(len(test_prob))])

    # Train the MIA shadow model (0 is forget set, 1 is test set)
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs"
    )
    clf.fit(X_mia, Y_mia)

    # Compute performance on forget set
    forget_res = clf.predict(X_forget)
    adv_forget_score = forget_res.mean() # Pr(A(m) = 1) for forget set (ideally low)

    # Compute performance on test set
    test_res = clf.predict(X_test)
    adv_test_score = test_res.mean() # Pr(A(m) = 1) for test set (ideally high)

    return adv_forget_score - adv_test_score


def run_swap_test(
    rf_model: nn.Module,
    rt_model: nn.Module,
    forget_ds: Dataset,
    test_ds: Dataset,
    batch_size: int,
    device: torch.device
) -> float:

    # s = (R, F, T) -> forgetting forget set
    adv_s = get_adv_advantage(
        forget_ds=forget_ds,
        test_ds=test_ds,
        candidate_model=rf_model,
        batch_size=batch_size,
        device=device
    )

    # s' = (R, T, F) -> forgetting test set
    adv_s_prime = get_adv_advantage(
        forget_ds=test_ds,
        test_ds=forget_ds,
        candidate_model=rt_model,
        batch_size=batch_size,
        device=device
    )

    # Advantage as computed by SWAP test
    swap_adv = np.abs(adv_s + adv_s_prime) / 2

    # Unlearning quality (should just be 1 - swap_adv)
    q_ul = 1 - swap_adv

    return q_ul
