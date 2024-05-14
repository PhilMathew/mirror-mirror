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
from distinguishers.random_perturbation_kld import compute_kld_over_perturbations
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise



def main():
    parser = ArgumentParser('Script for distinguishing via randomly adding noise then computing KLD')
    parser.add_argument('-d', '--ds_type', help='Dataset to use for training (one of MNIST, CIFAR10, or CIFAR100)')
    parser.add_argument('-p', '--perturbation_type', help='Type of perturbation to apply to the images') # TODO: Make README with options
    parser.add_argument('-f', '--forget_set_path', help='/path/to/file/defining/forget_set.csv')
    parser.add_argument('-om', '--original_checkpoint_path', help='/path/to/state/dict/for/original/model.pt')
    parser.add_argument('-u', '--unlearn_checkpoint_path', help='/path/to/state/dict/for/unlearned/model.pt')
    parser.add_argument('-c', '--control_checkpoint_path', help='/path/to/state/dict/for/control/model.pt')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/directory')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Number of training examples in a batch')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    match args.perturbation_type:
        case 'gaussian':
            perturbation_fn = lambda x: x + torch.normal(0, 0.01, x.shape)
        case _:
            raise ValueError(f'"{args.perturbation_type}" is an unknown type of image perturbation')
    
    # fig, axes = plt.subplots(1, 2 + len(perturbation_fns), figsize=(5 * len(perturbation_fns), 5))
    # CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    # orig_img = forget_ds[0][0].permute(1, 2, 0) * torch.Tensor(CIFAR_STD) + torch.Tensor(CIFAR_MEAN)
    # axes[0].imshow(orig_img)
    # axes[0].set_title('Original')
    # for i, (s, perturbation_fn) in enumerate(perturbation_fns.items()):
    #     img = perturbation_fn(orig_img)
    #     axes[i + 1].imshow(img / img.max())
    #     axes[i + 1].set_title(f'Mean 0 Variance {s}') 
    # axes[-1].imshow(torch.normal(0, 0.01, orig_img.shape))
    # axes[-1].set_title('Mean 0 Variance 0.01 HARDCODED')
    # fig.tight_layout()
    # fig.savefig('noise_test.png')
    
    print('Getting KLD score for unlearned model')
    unlearn_kld_score = compute_kld_over_perturbations(
        unlearned_model, 
        original_model,
        forget_ds,
        perturbation_fn,
        seed=0,
        device=device,
        batch_size=args.batch_size
    )
    
    print('Getting KLD score for control model')
    control_kld_score = compute_kld_over_perturbations(
        control_model, 
        original_model,
        forget_ds,
        perturbation_fn,
        seed=0,
        device=device,
        batch_size=args.batch_size
    )
    
    results_dict = {
        'unlearn_score': unlearn_kld_score,
        'control_score': control_kld_score,
        'outcome': 'unlearn' if unlearn_kld_score < control_kld_score else 'control'
    }
    results_df = pd.DataFrame({k: v if isinstance(v, list) else [v] for k, v in results_dict.items()})
    results_df.to_csv(str(output_dir / f'kld_random_{args.perturbation_type}_results.csv'), index=False)


if __name__ == '__main__':
    main()
