import json
from argparse import ArgumentParser
from pathlib import Path
from typing import *

import pandas as pd
import torch
from torch.utils.data import Subset

from components.datasets import init_full_ds
from components.resnet import build_resnet50
from distinguishers.logreg_mia import run_logreg_mia


def main():
    parser = ArgumentParser('Unlearning Script for M2')
    parser.add_argument('-d', '--ds_type', help='Dataset to use for training (one of MNIST, CIFAR10, or CIFAR100)')
    parser.add_argument('-mia', '--membership_inference_attack_type', help='Type of MIA to run') # TODO: Make README with options
    parser.add_argument('-f', '--forget_set_path', help='/path/to/file/defining/forget_set.csv')
    parser.add_argument('-u', '--unlearn_checkpoint_path', help='/path/to/state/dict/for/unlearned/model.pt')
    parser.add_argument('-c', '--control_checkpoint_path', help='/path/to/state/dict/for/control/model.pt')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/directory')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Number of training examples in a batch')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Deal with output path and save ther experiment arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(str(output_dir / f'{args.membership_inference_attack_type}_mia_args.json'), 'w') as f:
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
    unlearn_ckpt, control_ckpt = Path(args.unlearn_checkpoint_path), Path(args.control_checkpoint_path)
    unlearned_model, control_model = build_resnet50(num_classes, in_channels), build_resnet50(num_classes, in_channels)
    unlearned_model.load_state_dict(torch.load(str(unlearn_ckpt)))
    control_model.load_state_dict(torch.load(str(control_ckpt)))
    
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
    
    print('Getting MIA score for unlearned model')
    unlearn_mia_score, unlearn_num_pred_members = get_mia_score(unlearned_model)
    
    print('Getting MIA score for control model')
    control_mia_score, control_num_pred_members = get_mia_score(control_model)
    
    results_dict = {
        'unlearn_num_pred_members': unlearn_num_pred_members,
        'unlearn_score': unlearn_mia_score,
        'control_num_pred_members': control_num_pred_members,
        'control_score': control_mia_score,
        'outcome': 'unlearn' if unlearn_mia_score < control_mia_score else 'control'
    }
    results_df = pd.DataFrame({k: v if isinstance(v, list) else [v] for k, v in results_dict.items()})
    results_df.to_csv(str(output_dir / f'{args.membership_inference_attack_type}_mia_results.csv'), index=False)


if __name__ == '__main__':
    main()
