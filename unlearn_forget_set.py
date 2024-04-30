import json
from argparse import ArgumentParser
from pathlib import Path
from typing import *

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset
import numpy as np

from components.datasets import init_full_ds
from components.resnet import build_resnet50
from unlearning_frameworks.unlearning_methods import run_ssd
from utils.plot_utils import plot_confmat
from utils.train_utils import test_model


def main():
    parser = ArgumentParser('Unlearning Script for M2')
    parser.add_argument('-d', '--ds_type', help='Dataset to use for training (one of MNIST, CIFAR10, or CIFAR100)')
    parser.add_argument('-u', '--unlearning_method', help='Unlearning method to use') # TODO: Make README with options
    parser.add_argument('-f', '--forget_set_path', help='/path/to/file/defining/forget_set.csv')
    parser.add_argument('-ckpt', '--original_checkpoint_path', help='/path/to/state/dict/for/original/model.pt')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/directory')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Number of training examples in a batch')
    # parser.add_argument('--dampening_constant', type=float, default=1, help='Dampening constant for SSD unlearning method (only applicable if using SSD)')
    # parser.add_argument('--selection_weighting', type=float, default=10, help='Selection weighting for SSD unlearning method (only applicable if using SSD)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Deal with output path and save ther experiment arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(str(output_dir / 'unlearning_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Read in forget set
    forget_set_path = Path(args.forget_set_path)
    forget_set_df = pd.read_csv(str(forget_set_path))
    
    # Initialize full dataset and figure out classes and channels
    full_train_ds, test_ds, num_classes, in_channels = init_full_ds(args.ds_type)
    
    # Create forget set
    forget_inds = list(forget_set_df['sample_ind'])
    forget_ds = Subset(full_train_ds, indices=forget_inds)
    
    # Load the model
    original_ckpt = Path(args.original_checkpoint_path)
    original_model = build_resnet50(num_classes, in_channels)
    original_model.load_state_dict(torch.load(str(original_ckpt)))
    
    
        
    match args.unlearning_method:
        case 'ssd':
            dampening_constant, selection_weighting = 1, 5 # TODO: Document this somewhere
            unlearning_params = {'dampening_constant': dampening_constant, 'selection_weighting': None}
            unlearned_model = run_ssd(
                model=original_model,
                forget_ds=forget_ds,
                full_train_ds=full_train_ds,
                dampening_constant=dampening_constant,
                selection_weighting=selection_weighting,
                device=device,
                batch_size=args.batch_size
            )
        case _:
            raise ValueError(f'"{args.unlearning_method}" is an unknown unlearning method')
    
    # Save out M2 and unlearning params
    model_dir = output_dir / 'unlearn'
    model_dir.mkdir(exist_ok=True)
    torch.save(unlearned_model.state_dict(), str(model_dir / 'unlearn_state_dict.pt'))
    with open(str(model_dir / 'unlearning_params.json'), 'w') as f:
        json.dump(unlearning_params, f, indent=4)
    
    # Save out confusion matrix
    test_loss, test_acc, curr_preds, curr_labels = test_model(
        unlearned_model,
        test_ds, 
        device, 
        batch_size=args.batch_size, 
        return_preds_and_labels=True, 
        p_bar_desc='Testing unlearned model'
    )
    
    unlearned_model_confmat = confusion_matrix(curr_labels, curr_preds)
    plot_confmat(unlearned_model_confmat, save_path=str(model_dir / 'unlearn_confmat.png'), title=f'Loss: {test_loss:.4f}, Accuracy: {100 * test_acc:.4f}%')
    

if __name__ == '__main__':
    main()
