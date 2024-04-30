import json
from argparse import ArgumentParser
from pathlib import Path
from typing import *

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.resnet import Bottleneck
from tqdm import tqdm

from components.datasets import init_full_ds
from components.resnet import ResNet, build_resnet50
from utils.plot_utils import plot_confmat, plot_history
from utils.train_utils import test_model, train_model


def main():
    parser = ArgumentParser('Training script for M1 and M3')
    parser.add_argument('-d', '--ds_type', help='Dataset to use for training (one of MNIST, CIFAR10, or CIFAR100)')
    parser.add_argument('-f', '--forget_set_path', help='/path/to/file/defining/forget_set.csv')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/directory')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Number of training examples in a batch')
    parser.add_argument('-ne', '--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    args = parser.parse_args()
    
    # Deal with output path and save ther experiment arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(str(output_dir / 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Read in forget set
    forget_set_path = Path(args.forget_set_path)
    forget_set_df = pd.read_csv(str(forget_set_path))
    
    # Initialize full dataset and figure out classes and channels
    full_train_ds, test_ds, num_classes, in_channels = init_full_ds(args.ds_type, use_random_train_aug=True)
    
    # Create forget and retain sets
    forget_inds = list(forget_set_df['sample_ind'])
    forget_ds = Subset(full_train_ds, indices=forget_inds)
    retain_inds = [i for i in range(len(full_train_ds)) if i not in forget_inds]
    retain_ds = Subset(full_train_ds, indices=retain_inds)
    
    # Initialize and train original (full dataset) and control (only the retain set) model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_model, control_model = build_resnet50(num_classes, in_channels), build_resnet50(num_classes, in_channels)
    print('Training original model')
    original_train_hist = train_model(
        model=original_model,
        train_ds=full_train_ds,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.learning_rate
    )
    print('Training control model')
    control_train_hist = train_model(
        model=control_model,
        train_ds=retain_ds,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.learning_rate
    )
    
    # Save out M1-related stuff
    original_model_dir = output_dir / 'original'
    original_model_dir.mkdir(exist_ok=True)
    torch.save(original_model.state_dict(), str(original_model_dir / 'original_state_dict.pt')) # model
    with open(str(original_model_dir / 'original_train_hist.json'), 'w') as f: # train history
        json.dump(original_train_hist, f, indent=4)
    if args.num_epochs > 1: # plots are useless for training in a single epoch
        plot_history(original_train_hist, str(original_model_dir / 'original_train_hist.png')) # train history plots
    
    # Save out M3-related stuff
    control_model_dir = output_dir / 'm3'
    control_model_dir.mkdir(exist_ok=True)
    torch.save(control_model.state_dict(), str(control_model_dir / 'control_state_dict.pt')) # model
    with open(str(control_model_dir / 'control_train_hist.json'), 'w') as f: # train history
        json.dump(control_train_hist, f, indent=4)
    if args.num_epochs > 1: # plots are useless for training in a single epoch
        plot_history(control_train_hist, str(control_model_dir / 'control_train_hist.png')) # train history plots
    
    # Save confmats from testing
    original_test_loss, original_test_acc, original_preds, original_labels = test_model(
        original_model, 
        test_ds,
        device, 
        batch_size=args.batch_size,
        return_preds_and_labels=True, 
        p_bar_desc='Testing original model'
    )
    control_test_loss, control_test_acc, control_preds, control_labels = test_model(
        control_model, 
        test_ds, 
        device, 
        batch_size=args.batch_size, 
        return_preds_and_labels=True, 
        p_bar_desc='Testing control model'
    )
    original_confmat, control_confmat = confusion_matrix(original_labels, original_preds), confusion_matrix(control_labels, control_preds)
    plot_confmat(original_confmat, save_path=str(original_model_dir / 'original_confmat.png'), title=f'Loss: {original_test_loss:.4f}, Accuracy: {100 * original_test_acc:.4f}%')
    plot_confmat(control_confmat, save_path=str(control_model_dir / 'control_confmat.png'), title=f'Loss: {control_test_loss:.4f}, Accuracy: {100 * control_test_acc:.4f}%')


if __name__ == '__main__':
    main()
