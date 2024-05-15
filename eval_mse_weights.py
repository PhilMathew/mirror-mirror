import json
from argparse import ArgumentParser
from pathlib import Path
from typing import *

import pandas as pd
import torch
from torch.utils.data import Subset
import torchvision

from components.datasets import init_full_ds
from components.resnet import build_resnet50
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np



def main():
    parser = ArgumentParser('Script for distinguishing via randomly adding noise then computing KLD')
    parser.add_argument('-d', '--ds_type', help='Dataset to use for training (one of MNIST, CIFAR10, or CIFAR100)')
    parser.add_argument('-f', '--forget_set_path', help='/path/to/file/defining/forget_set.csv')
    parser.add_argument('-om', '--original_checkpoint_path', help='/path/to/state/dict/for/original/model.pt')
    parser.add_argument('-u', '--unlearn_checkpoint_path', help='/path/to/state/dict/for/unlearned/model.pt')
    parser.add_argument('-c', '--control_checkpoint_path', help='/path/to/state/dict/for/control/model.pt')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/directory')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Number of training examples in a batch')
    args = parser.parse_args()
    
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Deal with output path and save ther experiment arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    # with open(str(output_dir / f'{args.membership_inference_attack_type}_mia_args.json'), 'w') as f:
    #     json.dump(vars(args), f, indent=4)
    
    # Initialize full dataset and figure out classes and channels
    full_train_ds, test_ds, num_classes, in_channels = init_full_ds(args.ds_type)
    
    # Read in forget set
    forget_set_path = Path(args.forget_set_path)
    forget_set_df = pd.read_csv(str(forget_set_path))
    
    # Create forget and retain sets
    forget_inds = list(forget_set_df['sample_ind'])
    forget_ds = Subset(full_train_ds, indices=forget_inds)
    
    # Load M1, M2, M3
    original_ckpt, unlearn_ckpt, control_ckpt = Path(args.original_checkpoint_path), Path(args.unlearn_checkpoint_path), Path(args.control_checkpoint_path)
    original_model, unlearned_model, control_model = build_resnet50(num_classes, in_channels), build_resnet50(num_classes, in_channels), build_resnet50(num_classes, in_channels)
    original_model.load_state_dict(torch.load(str(original_ckpt)))
    unlearned_model.load_state_dict(torch.load(str(unlearn_ckpt)))
    control_model.load_state_dict(torch.load(str(control_ckpt)))

    unlearned_mse = torch.zeros(1)
    control_mse = torch.zeros(1)
    
    for name, _ in original_model.named_parameters():
        unlearned_mse += F.mse_loss(original_model.state_dict(name), unlearned_model.state_dict(name))
        control_mse += F.mse_loss(original_model.state_dict(name), control_model.state_dict(name))
    
    
    results_dict = {
        'unlearn_score': unlearned_mse,
        'control_score': control_mse,
        'outcome': 'unlearn' if unlearned_mse < control_mse else 'control'
    }
    results_df = pd.DataFrame({k: v if isinstance(v, list) else [v] for k, v in results_dict.items()})
    results_df.to_csv(str(output_dir / f'kld_random_{args.perturbation_type}_results.csv'), index=False)


if __name__ == '__main__':
    main()
