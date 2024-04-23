import json
from argparse import ArgumentParser
from pathlib import Path
from typing import *

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset

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
    parser.add_argument('-m1', '--m1_checkpoint_path', help='/path/to/state/dict/for/m1.pt')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/directory')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Number of training examples in a batch')
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
    m1_ckpt = Path(args.m1_checkpoint_path)
    m1 = build_resnet50(num_classes, in_channels)
    m1.load_state_dict(torch.load(str(m1_ckpt)))
    
    match args.unlearning_method:
        case 'ssd':
            m2 = run_ssd(
                model=m1,
                forget_ds=forget_ds,
                full_train_ds=full_train_ds,
                dampening_constant=1,
                selection_weighting=10,
                device=device,
                batch_size=args.batch_size
            )
        case _:
            raise ValueError(f'"{args.unlearning_method}" is an unknown unlearning method')
    
    # Save out M2
    m2_dir = output_dir / 'm2'
    m2_dir.mkdir(exist_ok=True)
    torch.save(m2.state_dict(), str(m2_dir / 'm2_state_dict.pt'))
    
    # Save test metrics for M2
    test_loss, test_acc, preds, labels = test_model(
        m2, 
        test_ds, 
        device, 
        batch_size=args.batch_size, 
        return_preds_and_labels=True, 
        p_bar_desc='Testing M2'
    )
    m2_confmat = confusion_matrix(labels, preds)
    plot_confmat(m2_confmat, save_path=str(m2_dir / 'm2_confmat.png'), title=f'Loss: {test_loss:.4f}, Accuracy: {100 * test_acc:.4f}%')
    

if __name__ == '__main__':
    main()
