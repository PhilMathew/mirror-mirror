from typing import *

import torch
from torch import nn
from torch.nn import functional as F


def calc_mse_score(
    candidate_model: nn.Module,
    original_model: nn.Module,
) -> float:
    mse_score = 0
    for name, _ in original_model.named_parameters():
        mse_score += F.mse_loss(original_model.state_dict()[name], candidate_model.state_dict()[name])
    
    return mse_score.item()
