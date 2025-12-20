import json
from argparse import ArgumentParser
from pathlib import Path
import random
from typing import *

import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.resnet import Bottleneck
from tqdm import tqdm
import plotly.express as px
import numpy as np
import copy

from components.datasets import init_full_ds
from components.models import build_model
from components.certified_removal import L2NormLayer
from distinguishers.logreg_mia import run_logreg_mia
from distinguishers.random_perturbation_kld import compute_kld_over_perturbations, compute_svm_consistency
from distinguishers.mse_score import calc_mse_score
from distinguishers.randomness_score import compute_model_randomness
from unlearning_frameworks.unlearning_methods import *
from utils.plot_utils import plot_confmat, plot_history
from utils.train_utils import test_model, train_model, add_cr_mechanism


def select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    return device


def load_experiment_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict


@torch.no_grad()
def find_mismatched_labels(test_dl, control_model, unlearned_model):
    mismatches = {k: [] for k in ('index', 'true_label', 'control_pred', 'unlearned_pred')}
    for i, (imgs, labels) in enumerate(tqdm(test_dl, desc='Finding mismatched labels')):
        imgs = imgs.cuda()
        control_preds = F.softmax(control_model(imgs), dim=-1).argmax(dim=-1).cpu()
        unlearned_preds = F.softmax(unlearned_model(imgs), dim=-1).argmax(dim=-1).cpu()
        for j in range(len(imgs)):
            if control_preds[j] != unlearned_preds[j]: # control model and unlearned model exhibit different behavior
                mismatches['index'].append(j)
                mismatches['true_label'].append(labels[j].item())
                mismatches['control_pred'].append(control_preds[j].item()) 
                mismatches['unlearned_pred'].append(unlearned_preds[j].item())
    print(f'Found {len(mismatches['index'])} mismatched labels')
    
    return mismatches


def main() -> None:
    parser = ArgumentParser('Script for producing motivating examples for overfitting')
    parser.add_argument('-d', '--model_dir', type=str, required=True, help='/path/to/model/directory')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/directory')
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    config_path = model_dir.parents[1] / 'experiment_config.yaml'
    config_dict = load_experiment_config(config_path)
    dataset_params = config_dict['dataset_params']
    train_params = config_dict['train_params']
    unlearning_methods = config_dict['unlearning_methods']
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = select_device()
    print(f'Using device {device}')
    
    # Initialize full dataset and figure out classes and channels
    full_train_ds, test_ds, num_classes, in_channels = init_full_ds(dataset_params['dataset_name'])
    test_dl = DataLoader(test_ds, batch_size=config_dict['batch_size'], num_workers=train_params['num_workers'])
    
    # Construct the forget set
    forget_set_df = pd.read_csv(str(model_dir / 'forget_set.csv'))
    forget_inds = list(forget_set_df['sample_ind'])
    forget_ds = Subset(full_train_ds, indices=forget_inds)
    
    # Load the control model
    control_state_dict_path = model_dir / 'control' / 'control_state_dict.pt'
    control_model =  build_model(
        model_type=train_params['model_type'],
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=False,
        use_differential_privacy=train_params.get('use_differential_privacy', False)
    )
    control_model.load_state_dict(torch.load(str(control_state_dict_path)))
    control_model.eval()
    control_model.to(device)
    
    for unlearning_method in unlearning_methods:
        if unlearning_method != 'control': # don't need to compare control to itself
            # Load in the model
            unlearned_state_dict_path = model_dir / unlearning_method / f'{unlearning_method}_state_dict.pt'
            unlearned_model =  build_model(
                model_type=train_params['model_type'],
                num_classes=num_classes,
                in_channels=in_channels,
                pretrained=False,
                use_differential_privacy=train_params.get('use_differential_privacy', False)
            )
            unlearned_model.load_state_dict(torch.load(str(unlearned_state_dict_path)))
            unlearned_model.eval()
            unlearned_model.to(device)
            
            # Find mismatched labels
            mismatches = find_mismatched_labels(test_dl, control_model, unlearned_model)
            mismatches_df = pd.DataFrame(mismatches)
            mismatches_df.to_csv(output_dir / f'{unlearning_method}_mismatches.csv', index=False)


if __name__ == "__main__":
    main()
