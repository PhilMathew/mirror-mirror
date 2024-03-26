from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from typing import *
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from plotly import express as px
import json
import numpy as np

from components.datasets import init_full_ds


def main():
    parser = ArgumentParser('Script to generate forget set')
    parser.add_argument('-d', '--ds_type', help='Dataset to use for training (one of MNIST, CIFAR10, or CIFAR100)')
    parser.add_argument(
        '-c', '--class_to_forget', type=str,
        help='Class to remove as the forget set (if "random" then a random number of examples are selected without attention given to their classification)'
    )
    parser.add_argument('-n', '--num_forget', type=int, help='Number of examples to forget')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/directory')
    args = parser.parse_args()
    
    # Deal with output path and save ther experiment arguments
    output_dir = Path(args.output_dir) / 'datasets'
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(str(output_dir / 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize the full dataset
    train_ds, _ = init_full_ds(args.ds_type)
    
    # Aggregate indices and labels in forget set
    forget_set_dict = {'sample_ind': [], 'label': []}
    if args.class_to_forget == 'random':
        forget_inds = random.sample(range(len(train_ds)), k=args.num_forget)
    else:
        forget_inds = [i for i in range(len(train_ds)) if train_ds[i][1] == int(args.class_to_forget)][:args.num_forget]
    for i in tqdm(forget_inds, desc='Generating Forget Set'):
        forget_set_dict['sample_ind'].append(i)
        forget_set_dict['label'].append(train_ds[i][1])
        
    # Save forget set as a CSV
    forget_set_df = pd.DataFrame(forget_set_dict)
    forget_set_df.to_csv(str(output_dir / 'forget_set.csv'), index=False)
    
    # Save forget set distribution
    forget_set_df['label'] = forget_set_df['label'].apply(str)
    counts = np.bincount(list(forget_set_df['label']))
    labels = [str(i) for i in range(len(counts))]
    fig = px.bar(
        x=labels,
        y=counts,
        color=labels,
        labels={
            'x': 'Label',
            'y': 'Count',
            'color': 'Label'
        },
        title='Forget Set Label Distribution'
    )
    fig.write_image(str(output_dir / 'forget_set_distribution.png'), format='png')


if __name__ == '__main__':
    main()
