import copy
from argparse import ArgumentParser
from pathlib import Path
from typing import *

import pandas as pd
import torch
import yaml
from torch import nn
import numpy as np
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from components.certified_removal import L2NormLayer
from sklearn.metrics import roc_curve
from components.datasets import init_full_ds
from components.models import build_model
from run_experiment import (compute_distinguisher_score,
                            eval_model_performance, load_experiment_config,
                            run_unlearning, select_device, train_single_model)
from utils.train_utils import add_cr_mechanism


def get_clean_and_poisoned_evals(    
    model: nn.Module,
    clean_test_ds: Dataset,
    poisoned_test_ds: Dataset,
    device: torch.device,
    batch_size: int,
    model_dir: Path,
    num_workers: int,
    p_bar_desc: str,
):
    # Clean performance
    clean_loss, clean_acc, _, _, _ = eval_model_performance(
        model=model,
        test_ds=clean_test_ds,
        device=device,
        batch_size=batch_size,
        model_dir=model_dir / 'clean',
        num_workers=num_workers,
        p_bar_desc=p_bar_desc
    )
    
    # Poisoned performance
    poison_loss, poison_acc, poison_pred_probs, _, poison_labels = eval_model_performance(
        model=model,
        test_ds=poisoned_test_ds,
        device=device,
        batch_size=batch_size,
        model_dir=model_dir / 'poisoned',
        num_workers=num_workers,
        p_bar_desc=p_bar_desc
    )
    
    # Get ROC curve
    fpr, tpr, thresholds = roc_curve(poison_labels, poison_pred_probs[:, 0], pos_label=0) # mapping poisoned examples to label 0, thus making it the positive label
    tpr_at_low_fpr = tpr[np.argmin(np.abs(fpr - 0.01))] # finds the index of the closest FPR to 0.01 and returns the corresponding TPR
    # breakpoint()
    return clean_loss, clean_acc, poison_loss, poison_acc, tpr_at_low_fpr


def create_original_model(
    train_params: Dict,
    config_dict: Dict,
    unlearning_methods: Dict,
    output_dir: Path,
    full_train_ds_random_aug: Dataset,
    use_existing_models: bool,
    full_train_ds: Dataset,
    num_classes: int,
    in_channels: int,
    device: torch.device
):
    original_state_dict_path = output_dir / 'original' / 'original_state_dict.pt'
    if use_existing_models and original_state_dict_path.exists():
        original_model = build_model(
            model_type=train_params['model_type'],
            num_classes=num_classes, 
            in_channels=in_channels,
            pretrained=False, # we're loading in weights anyway
            use_differential_privacy=train_params.get('use_differential_privacy', False)
        )
        original_model.load_state_dict(torch.load(str(original_state_dict_path)))
        original_model = original_model.to(device)
        print(f'Using pre-trained original model from {original_state_dict_path}')
    else:
        # Train the original model (M1) on the whole dataset
        original_model = train_single_model(
            train_ds=full_train_ds_random_aug,
            num_classes=num_classes,
            in_channels=in_channels,
            batch_size=config_dict['batch_size'],
            device=device,
            model_name='original',
            output_dir=output_dir,
            **train_params
        )
    
    # Add a certified removal mechanism if necessary
    if train_params.get('use_certified_removal', False): 
        assert train_params['use_differential_privacy'], 'Certified removal requires a DP model as the feature extractor'
        assert 'resnet' in train_params['model_type'], 'Certified removal only works with ResNet models'
        original_state_dict_path = output_dir / 'original' / f'original_cr_state_dict.pt'
        
        print('Adding certified removal mechanism to the original model')
        print('NOTE: THIS WILL NOT PLAY NICE IF OTHER UNLEARNING METHODS ARE USED')
        if use_existing_models and original_state_dict_path.exists():
            original_model = build_model(
                model_type=train_params['model_type'],
                num_classes=num_classes, 
                in_channels=in_channels,
                pretrained=False, # we're loading in weights anyway
                use_differential_privacy=train_params.get('use_differential_privacy', False)
            )
            
            original_model.prefc_norm = L2NormLayer() # CR uses an L2 norm prior to the final layer
            original_model.fc = nn.Linear(original_model.fc.in_features, num_classes, bias=False) # re-initialize the final layer but remove the bias term
            original_model.load_state_dict(torch.load(str(original_state_dict_path)))
            original_model = original_model.to(device)
        else:
            original_model = add_cr_mechanism(
                model=original_model,
                train_ds=full_train_ds,
                lam=unlearning_methods['certified_removal']['lambda'],
                sigma=unlearning_methods['certified_removal']['sigma'],
                batch_size=config_dict['batch_size'],
                num_workers=train_params['num_workers'],
                device=device
            )
            torch.save(original_model.state_dict(), str(original_state_dict_path)) # model
        print('Certified removal mechanism added successfully')
    
    return original_model, original_state_dict_path





def main():
    parser = ArgumentParser('Script to run an unlearning experiment')
    parser.add_argument('-c', '--config_path', help='/path/to/experiment/config.yaml') # TODO: Make an example YAML
    parser.add_argument('-o', '--output_dir', default='./output', help='/path/to/output/directory')
    parser.add_argument('--use_existing_models', action='store_true', help='Uses trained models from a previous experiment')
    parser.add_argument('--redo_unlearning', action='store_true', help='Re-runs the unlearning methods only')
    args = parser.parse_args()
    
    # Load in the config
    config_path = Path(args.config_path)
    config_dict = load_experiment_config(config_path)
    dataset_params = config_dict['dataset_params']
    train_params = config_dict['train_params']
    unlearning_methods = config_dict['unlearning_methods']
    distinguisher_params = config_dict['distinguisher_params']
    
    # Deal with output path and save the copy the experiment config in it
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(str(output_dir / f'experiment_config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Select the device to use
    device = select_device()
    print(f'Using device {device}')
    
    # Iterate over all the desired forget sets
    score_dicts = {d: {k: [] for k in ('forget_set', 'run_number', 'model', 'score')} for d in ('clean_accuracy', 'poison_accuracy', 'poison_tpr', *distinguisher_params.keys())}
    for forget_set_ind, poison_rate in enumerate(dataset_params['poison_rate']):
        if poison_rate >= 1.0:
            raise ValueError("Poison rate must be less than 1.0 (otherwise retain set is empty)")
        
        for run_num in range(config_dict['runs_per_forget_set']):
            forget_set_name = f'forget_poison_{poison_rate}'
            print(f'Poison Rate: {poison_rate}, Run number: {run_num}')
            
            # Create the directory to store everything under
            curr_output_dir = output_dir / forget_set_name / f'run_{run_num}'
            curr_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize datasets
            full_train_ds, poison_test_ds, num_classes, in_channels = init_full_ds(dataset_params['dataset_name'], do_poison=True, poison_rate=poison_rate, poison_type=dataset_params['poison_type'])
            # We make a second copy of the training set with random augmentations applied, which we use for training
            full_train_ds_random_aug, _, _, _ = init_full_ds(dataset_params['dataset_name'], use_random_train_aug=True, do_poison=True, poison_rate=poison_rate, poison_type=dataset_params['poison_type'])
            # Also want a copy of the clean test set
            _, clean_test_ds, _, _ = init_full_ds(dataset_params['dataset_name'], do_poison=False)
            
            # Construct the forget set from the poisoned examples
            # First, if we're using a previous experiment, make sure the forget set is the same
            if args.use_existing_models and (curr_output_dir / 'forget_set.csv').exists():
                poisoned_set_df = pd.read_csv(str(curr_output_dir / 'forget_set.csv'))
                poisoned_inds = list(poisoned_set_df['sample_ind'])
                full_train_ds.poisoned_set = frozenset(poisoned_inds)
                
            # Then, make sure the poisoned sets are the same across the two datasets
            full_train_ds_random_aug.poisoned_set = copy.deepcopy(full_train_ds.poisoned_set) # may be unnecessary but I'm not taking chances
            forget_inds = list(full_train_ds.poisoned_set)
            forget_ds = Subset(full_train_ds, indices=forget_inds)
            
            # Have to save things out here if needed
            if not (curr_output_dir / 'forget_set.csv').exists():
                forget_set_df = pd.DataFrame({'sample_ind': forget_inds, 'label': [full_train_ds[i][1] for i in forget_inds]})
                forget_set_df.to_csv(str(curr_output_dir / 'forget_set.csv'), index=False) 
            
            original_model, original_state_dict_path = create_original_model(
                train_params=train_params,
                config_dict=config_dict,
                unlearning_methods=unlearning_methods,
                output_dir=curr_output_dir,
                full_train_ds=full_train_ds,
                full_train_ds_random_aug=full_train_ds_random_aug,
                use_existing_models=args.use_existing_models,
                num_classes=num_classes,
                in_channels=in_channels,
                device=device
            )
            
            get_clean_and_poisoned_evals(
                model=original_model,
                clean_test_ds=clean_test_ds,
                poisoned_test_ds=poison_test_ds,
                device=device,
                batch_size=config_dict['batch_size'],
                model_dir=curr_output_dir / 'original',
                num_workers=train_params['num_workers'],
                p_bar_desc='Testing original model'
            )
            
            # Create copies of retain set with and without random augmentations
            retain_inds = [i for i in range(len(full_train_ds)) if i not in forget_inds]
            retain_ds = Subset(full_train_ds, indices=retain_inds)
            retain_ds_random_aug = Subset(full_train_ds_random_aug, indices=retain_inds)
            
            # Train the control model
            if args.use_existing_models and (curr_output_dir / 'control' / 'control_state_dict.pt').exists():
                control_model = build_model(
                    model_type=train_params['model_type'],
                    num_classes=num_classes, 
                    in_channels=in_channels,
                    pretrained=False, # we're loading in weights anyway
                    use_differential_privacy=train_params.get('use_differential_privacy', False)
                )
                control_model.load_state_dict(torch.load(str(curr_output_dir / 'control' / 'control_state_dict.pt'), weights_only=True))
                control_model = control_model.to(device)
            else:
                # if train_params['use_certified_removal']:
                #     assert train_params['use_differential_privacy'], 'Certified removal experiments must compare to a DP control model'
                control_model = train_single_model(
                    train_ds=retain_ds_random_aug,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    batch_size=config_dict['batch_size'],
                    device=device,
                    model_name='control',
                    output_dir=curr_output_dir,
                    **train_params
                )
                
                # get_clean_and_poisoned_evals(
                #     model=control_model,
                #     clean_test_ds=clean_test_ds,
                #     poisoned_test_ds=poison_test_ds,
                #     device=device,
                #     batch_size=config_dict['batch_size'],
                #     model_dir=curr_output_dir / 'control',
                #     num_workers=train_params['num_workers'],
                #     p_bar_desc='Testing control model'
                # )
            
            models_to_score = {'control': control_model.cpu()}
            # Get unlearned models for each unlearning method
            for unlearning_method_key, unlearning_params in unlearning_methods.items():
                unlearning_method = unlearning_method_key.split('.')[0]
                unlearned_model_dir = curr_output_dir / unlearning_method_key
                unlearned_model_ckpt = unlearned_model_dir / f'{unlearning_method}_state_dict.pt'
                print(f'Running {unlearning_method_key}')
                if not args.redo_unlearning and args.use_existing_models and unlearned_model_ckpt.exists():
                    unlearned_model = build_model(
                        model_type=train_params['model_type'],
                        num_classes=num_classes, 
                        in_channels=in_channels,
                        pretrained=False, # we're loading in weights anyway
                        use_differential_privacy=train_params.get('use_differential_privacy', False)
                    )
                    if unlearning_method == 'certified_removal':
                        unlearned_model.fc = nn.Linear(unlearned_model.fc.in_features, num_classes, bias=False) # re-initialize the final layer but remove the bias term
                    unlearned_model.load_state_dict(torch.load(str(unlearned_model_ckpt), weights_only=True))
                    unlearned_model.to(device)
                else:
                    if unlearning_method == 'dp_sgd_retrained': # in case of DP-SGD we unlearn by just making a new DP model
                        curr_seed = torch.seed()
                        torch.manual_seed(curr_seed + 1)
                        unlearned_model = train_single_model(
                            train_ds=full_train_ds_random_aug,
                            num_classes=num_classes,
                            in_channels=in_channels,
                            batch_size=config_dict['batch_size'],
                            device=device,
                            model_name='dp_sgd_retrained',
                            output_dir=curr_output_dir,
                            **train_params
                        )
                    else:
                        unlearned_model = run_unlearning(
                            unlearning_method=unlearning_method,
                            unlearning_params=unlearning_params,
                            original_model=original_model,
                            forget_ds=forget_ds,
                            forget_inds=forget_inds,
                            full_train_ds=full_train_ds,
                            retain_ds=retain_ds,
                            num_classes=num_classes,
                            in_channels=in_channels,
                            batch_size=config_dict['batch_size'],
                            device=device,
                            output_dir=unlearned_model_dir,
                            **train_params,
                        )
                    
                    # get_clean_and_poisoned_evals(
                    #     model=unlearned_model,
                    #     clean_test_ds=clean_test_ds,
                    #     poisoned_test_ds=poison_test_ds,
                    #     device=device,
                    #     batch_size=config_dict['batch_size'],
                    #     model_dir=unlearned_model_dir,
                    #     num_workers=train_params['num_workers'],
                    #     p_bar_desc='Testing unlearned model'
                    # )
                
                # Load the model onto CPU
                models_to_score[unlearning_method_key] = unlearned_model
                
                # We have to reload the original model the unlearning methods appear to act in place
                original_model.load_state_dict(torch.load(str(original_state_dict_path), weights_only=True))
            
            
            
            # Get distinguisher scores for the models
            for distinguisher, curr_distinguisher_params in distinguisher_params.items():
                for model_name, model in tqdm(models_to_score.items(), desc=f'Scoring models for {distinguisher}'):
                    score = compute_distinguisher_score(
                        distinguisher=distinguisher,
                        candidate_model=model,
                        original_model=original_model,
                        num_classes=num_classes,
                        forget_ds=forget_ds,
                        retain_ds=retain_ds,
                        test_ds=poison_test_ds,
                        batch_size=config_dict['batch_size'],
                        device=device,
                        distinguisher_params=curr_distinguisher_params,
                        num_workers=train_params['num_workers'],
                    )
                    score_dicts[distinguisher]['forget_set'].append(forget_set_name[len("forget_"):])
                    score_dicts[distinguisher]['run_number'].append(run_num)
                    score_dicts[distinguisher]['model'].append(model_name)
                    score_dicts[distinguisher]['score'].append(score)

            # Add in clean and poisoned performance
            for model_name, model in tqdm(models_to_score.items(), desc=f'Scoring models for {distinguisher}'):
                _, clean_acc, _, poison_acc, poison_tpr = get_clean_and_poisoned_evals( # this also saves out the performance of the model
                    model=model,
                    clean_test_ds=clean_test_ds,
                    poisoned_test_ds=poison_test_ds,
                    device=device,
                    batch_size=config_dict['batch_size'],
                    model_dir=curr_output_dir / model_name,
                    num_workers=train_params['num_workers'],
                    p_bar_desc=f'Testing {model_name} model'
                )
                
                score_dicts['clean_accuracy']['forget_set'].append(forget_set_name[len("forget_"):])
                score_dicts['clean_accuracy']['run_number'].append(run_num)
                score_dicts['clean_accuracy']['model'].append(model_name)
                score_dicts['clean_accuracy']['score'].append(clean_acc)
                
                score_dicts['poison_accuracy']['forget_set'].append(forget_set_name[len("forget_"):])
                score_dicts['poison_accuracy']['run_number'].append(run_num)
                score_dicts['poison_accuracy']['model'].append(model_name)
                score_dicts['poison_accuracy']['score'].append(poison_acc)
                
                score_dicts['poison_tpr']['forget_set'].append(forget_set_name[len("forget_"):])
                score_dicts['poison_tpr']['run_number'].append(run_num)
                score_dicts['poison_tpr']['model'].append(model_name)
                score_dicts['poison_tpr']['score'].append(poison_tpr)
            
            
            # Save the resulting scores from each distinguisher into a separate CSV
            for distinguisher, score_dict in score_dicts.items():
                print('Saving out the results')
                score_df = pd.DataFrame(score_dict)
                score_df.to_csv(str(output_dir / f'{distinguisher}_results.csv'), index=False)


if __name__ == "__main__":
    main()
