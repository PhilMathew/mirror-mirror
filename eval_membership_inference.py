import json
from argparse import ArgumentParser
from pathlib import Path
from typing import *

import pandas as pd
import torch
from torch.utils.data import Subset

from components.datasets import init_full_ds
from components.resnet import build_resnet50
from membership_inference.logreg_mia import run_logreg_mia


def main():
    parser = ArgumentParser('Unlearning Script for M2')
    parser.add_argument('-d', '--ds_type', help='Dataset to use for training (one of MNIST, CIFAR10, or CIFAR100)')
    parser.add_argument('-mia', '--membership_inference_attack_type', help='Type of MIA to run') # TODO: Make README with options
    parser.add_argument('-f', '--forget_set_path', help='/path/to/file/defining/forget_set.csv')
    parser.add_argument('-m2', '--m2_checkpoint_path', help='/path/to/state/dict/for/m2.pt')
    parser.add_argument('-m3', '--m3_checkpoint_path', help='/path/to/state/dict/for/m3.pt')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/directory')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Number of training examples in a batch')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Deal with output path and save ther experiment arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(str(output_dir / 'mia_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize full dataset and figure out classes and channels
    full_train_ds, test_ds, num_classes, in_channels = init_full_ds(args.ds_type)
    
    # Read in forget set
    forget_set_path = Path(args.forget_set_path)
    forget_set_df = pd.read_csv(str(forget_set_path))
    
    # Create forget and retain sets
    forget_inds = list(forget_set_df['sample_ind'])
    forget_ds = Subset(full_train_ds, indices=forget_inds)
    retain_inds = [i for i in range(len(full_train_ds)) if i not in forget_inds]
    retain_ds = Subset(full_train_ds, indices=retain_inds) # don't need to worry about the test dataset since it contains no members obviously
    
    # Load M2, M3
    m2_ckpt, m3_ckpt = Path(args.m2_checkpoint_path), Path(args.m3_checkpoint_path)
    m2, m3 = build_resnet50(num_classes, in_channels), build_resnet50(num_classes, in_channels)
    m2.load_state_dict(torch.load(str(m2_ckpt)))
    m3.load_state_dict(torch.load(str(m3_ckpt)))
    
    match args.membership_inference_attack_type:
        case 'logreg':
            get_mia_score = lambda model: run_logreg_mia(
                model=model,
                model_forget_ds=forget_ds,
                model_retain_ds=retain_ds,
                model_test_ds=test_ds,
                batch_size=args.batch_size,
                device=device
            )
        case _:
            raise ValueError(f'"{args.membership_inference_attack_type}" is an unknown type of membership inference attack')
    
    print('Getting MIA score for M2')
    m2_mia_score, m2_num_pred_members = get_mia_score(m2)
    
    print('Getting MIA score for M3')
    m3_mia_score, m3_num_pred_members = get_mia_score(m3)
    
    results_dict = {
        'num_actual_members': 0, # running on the forget set os obviously this is 0
        'm2': {
            'ckpt': args.m2_checkpoint_path,
            'num_pred_members': m2_num_pred_members,
            f'{args.membership_inference_attack_type}_score': m2_mia_score
        },
        'm3': {
            'ckpt': args.m3_checkpoint_path,
            'num_pred_members': m3_num_pred_members,
            f'{args.membership_inference_attack_type}_score': m3_mia_score
        },
        'outcome': 'M2' if m2_mia_score > m3_mia_score else 'M3'
    }
    
    with open(str(output_dir / f'{args.membership_inference_attack_type}_mia_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)


if __name__ == '__main__':
    main()
