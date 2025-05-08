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

from mirrormirror.components.datasets import init_full_ds
from mirrormirror.components.resnet import build_resnet18
from mirrormirror.components.certified_removal import L2NormLayer
from mirrormirror.distinguishers.logreg_mia import run_logreg_mia
from mirrormirror.distinguishers.random_perturbation_kld import compute_kld_over_perturbations, compute_svm_consistency
from mirrormirror.distinguishers.mse_score import calc_mse_score
from mirrormirror.distinguishers.randomness_score import compute_model_randomness
from mirrormirror.unlearning_frameworks.unlearning_methods import *
from mirrormirror.utils.plot_utils import plot_confmat, plot_history
from mirrormirror.utils.train_utils import test_model, train_model, add_cr_mechanism


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


def generate_forget_set(
    full_ds: Dataset,
    num_classes: int,
    class_to_forget: str,
    output_dir: Path,
    num_forget: Optional[int] = None,
) -> Tuple[Dataset, List[int]]:
    # Aggregate indices and labels in forget set
    forget_set_dict = {'sample_ind': [], 'label': []}
    if class_to_forget == 'random':
        assert num_forget is not None, 'num_forget must be specified for randomly selected forget set'
        forget_inds = random.sample(range(len(full_ds)), k=num_forget)
    else:
        forget_inds = [i for i in range(len(full_ds)) if full_ds[i][1] == int(class_to_forget)]
        if num_forget is not None:
            forget_inds = forget_inds[:num_forget]

    for i in tqdm(forget_inds, desc='Generating Forget Set'):
        forget_set_dict['sample_ind'].append(i)
        forget_set_dict['label'].append(full_ds[i][1])
    
    # Save forget set as a CSV
    forget_set_df = pd.DataFrame(forget_set_dict)
    forget_set_df.to_csv(str(output_dir / 'forget_set.csv'), index=False)
    
    # Save forget set distribution
    forget_set_df['label'] = forget_set_df['label'].apply(str)
    labels = [str(i) for i in range(num_classes)]
    counts = np.bincount(list(forget_set_df['label']), minlength=len(labels))
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
    fig.write_html(str(output_dir / 'forget_set_distribution.html'))
    
    # Create forget set
    forget_inds = list(forget_set_df['sample_ind'])
    forget_ds = Subset(full_ds, indices=forget_inds)
    
    return forget_ds, forget_inds


def eval_model_performance(
    model: nn.Module,
    test_ds: Dataset,
    device: torch.device,
    batch_size: int,
    model_dir: Path,
    num_workers: int,
    p_bar_desc: str,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    
    test_loss, test_acc, curr_preds, curr_labels = test_model(
        model,
        test_ds, 
        device, 
        batch_size=batch_size, 
        return_preds_and_labels=True, 
        p_bar_desc=p_bar_desc,
        num_workers = num_workers,
    )
    
    model_confmat = confusion_matrix(curr_labels, curr_preds)
    plot_confmat(
        model_confmat, 
        save_path=str(model_dir / 'confmat.png'), 
        title=f'Loss: {test_loss:.4f}, Accuracy: {100 * test_acc:.4f}%'
    )


def train_single_model(
    train_ds: Dataset,
    test_ds: Dataset,
    num_classes: int,
    in_channels: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    model_name: str,
    output_dir: Path,
    use_differential_privacy: bool = False,
    **kwargs
) -> nn.Module:
    model = build_resnet18(
        num_classes, 
        in_channels, 
        pretrained=kwargs.get('pretrained', True), 
        use_differential_privacy=use_differential_privacy
    )
    
    print(f'Training {model_name} model')   
    train_hist = train_model(
        model=model,
        train_ds=train_ds,
        device=device,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=learning_rate,
        use_differential_privacy=use_differential_privacy,
        **kwargs
    ) 
    
    # Save out training history and model state dict
    model_dir = output_dir / model_name
    model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), str(model_dir / f'{model_name}_state_dict.pt')) # model
    with open(str(model_dir / f'{model_name}_train_hist.json'), 'w') as f: # train history
        json.dump(train_hist, f, indent=4)
    if num_epochs > 1: # plots are useless for training in a single epoch
        plot_history(train_hist, str(model_dir / f'{model_name}_train_hist.png')) # train history plots
    num_workers = kwargs.get('num_workers', 16)
    # Evaluate performance
    eval_model_performance(model, test_ds, device, batch_size, model_dir, num_workers, p_bar_desc=f'Testing {model_name} model')
    
    return model


def run_unlearning(
    unlearning_method: str,
    unlearning_params: dict,
    original_model: nn.Module,
    forget_ds: Dataset,
    forget_inds: List[int],
    full_train_ds: Dataset,
    retain_ds: Dataset,
    test_ds: Dataset,
    num_classes: int,
    in_channels: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    num_workers: int,
    **kwargs
) -> nn.Module:
    # To ensure the original model is not modified during unlearning, we create a copy
    orig_model_weights = copy.deepcopy(original_model.state_dict())
    original_model = build_resnet18(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=False,
        use_differential_privacy=kwargs.get('use_differential_privacy', False)
    )
    if unlearning_method == 'certified_removal':
        original_model.prefc_norm = L2NormLayer()
        original_model.fc = nn.Linear(original_model.fc.in_features, original_model.fc.out_features, bias=False)
    original_model.load_state_dict(orig_model_weights)
    match unlearning_method:
        case 'ssd':
            unlearned_model = run_ssd(
                model=original_model,
                forget_ds=forget_ds,
                full_train_ds=full_train_ds,
                dampening_constant=unlearning_params['dampening_constant'],
                selection_weighting=unlearning_params['selection_weighting'],
                device=device,
                batch_size=batch_size,
                num_workers=num_workers
            )
        case 'fisher_forgetting':
            unlearned_model = run_fisher_forgetting(
                model=original_model,
                retain_ds=retain_ds,
                num_classes=num_classes,
                device=device
            )
        case 'amnesiac':
            unlearned_model = run_amnesiac(
                model=original_model,
                forget_ds=forget_ds,
                retain_ds=retain_ds,
                device=device,
                num_classes=num_classes,
                forget_class=forget_ds[0][-1]
            )
        case 'bad_teacher':
            unlearning_teacher = build_resnet18(num_classes, in_channels).to(device)
            unlearned_model = run_bad_teacher(
                model=original_model,
                unlearning_teacher=unlearning_teacher,
                forget_ds=forget_ds,
                retain_ds=retain_ds,
                device=device,
                
            )
        case 'certified_removal':
            unlearned_model = run_certified_removal(
                model=original_model,
                full_train_ds=full_train_ds,
                forget_inds=forget_inds,
                lam=unlearning_params['lambda'],
                device=device,
                batch_size=batch_size,
                num_workers=num_workers
            )
        case 'certified_deep_unlearning':
            unlearned_model = run_certified_deep_unlearning(
                model=original_model,
                retain_ds=retain_ds,
                weight_decay=unlearning_params['weight_decay'],
                s1=unlearning_params['s1'],
                s2=unlearning_params['s2'],
                gamma=unlearning_params['gamma'],
                scale=unlearning_params['scale'],
                std=unlearning_params['std'],
                device=device,
                batch_size=batch_size,
                num_workers=num_workers
            ) 
        case _:
            raise ValueError(f'"{unlearning_method}" is an unknown unlearning method')
    unlearned_model_dir = output_dir    
    unlearned_model_dir.mkdir(exist_ok=True)
    torch.save(unlearned_model.state_dict(), str(unlearned_model_dir / f'{unlearning_method}_state_dict.pt'))
    eval_model_performance(
        model=unlearned_model,
        test_ds=test_ds,
        device=device,
        batch_size=batch_size,
        model_dir=unlearned_model_dir,
        num_workers=num_workers,
        p_bar_desc=f'Testing unlearned model ({unlearning_method})'
    )
    
    return unlearned_model


def compute_distinguisher_score(
    distinguisher: str,
    candidate_model: nn.Module,
    original_model: nn.Module,
    forget_ds: Dataset,
    retain_ds: Dataset,
    test_ds: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    distinguisher_params: dict,
) -> float:
    match distinguisher:
        case 'mia':
            match distinguisher_params['mia_type']:
                case 'logreg':
                    distinguisher_score = run_logreg_mia(
                        candidate_model=candidate_model,
                        original_model=original_model,
                        model_forget_ds=forget_ds,
                        model_retain_ds=retain_ds,
                        model_test_ds=test_ds,
                        batch_size=batch_size,
                        device=device
                    )
                case _:
                    raise ValueError(f'"{distinguisher_params["mia_type"]}" is an unknown type of membership inference attack')
        case 'kld':
            match distinguisher_params['perturbation_type']:
                case 'gaussian':
                    perturbation_fn = lambda x: x + torch.normal(0, 0.1, x.shape)
                case 'no_perturbation':
                    perturbation_fn = lambda x: x
                case _:
                    raise ValueError(f'"{distinguisher_params["perturbation_type"]}" is an unknown type of image perturbation')
            distinguisher_score = compute_kld_over_perturbations(
                candidate_model=candidate_model, 
                original_model=original_model,
                dataset=forget_ds,
                perturbation_fn=perturbation_fn,
                seed=0,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers
            )
        case 'consistency':
            match distinguisher_params['perturbation_type']:
                case 'gaussian':
                    perturbation_fn = lambda x: x + torch.normal(0, 0.1, x.shape)
                case 'no_perturbation':
                    perturbation_fn = lambda x: x
                case _:
                    raise ValueError(f'"{distinguisher_params["perturbation_type"]}" is an unknown type of image perturbation')
            distinguisher_score = compute_svm_consistency(
                candidate_model=candidate_model, 
                original_model=original_model,
                dataset=forget_ds,
                perturbation_fn=perturbation_fn,
                seed=0,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers
            )
        case 'mse':
            distinguisher_score = calc_mse_score(
                candidate_model=candidate_model,
                original_model=original_model
            )
        case 'randomness':
            match distinguisher_params['perturbation_type']:
                case 'gaussian':
                    perturbation_fn = lambda x: x + torch.normal(0, 0.1, x.shape)
                case 'no_perturbation':
                    perturbation_fn = lambda x: x
                case _:
                    raise ValueError(f'"{distinguisher_params["perturbation_type"]}" is an unknown type of image perturbation')
            distinguisher_score = compute_model_randomness(
                candidate_model=candidate_model, 
                original_model=original_model,
                dataset=forget_ds,
                perturbation_fn=perturbation_fn,
                seed=0,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers
            )
        case _:
            raise ValueError(f'"{distinguisher}" is an unknown type of distinguisher')
    
    return distinguisher_score


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
    
    # Initialize full dataset and figure out classes and channels
    full_train_ds, test_ds, num_classes, in_channels = init_full_ds(dataset_params['dataset_name'])
    # We make a second copy of the training set with random augmentations applied, which we use for training
    full_train_ds_random_aug, _, _, _ = init_full_ds(dataset_params['dataset_name'], use_random_train_aug=True)
    original_state_dict_path = output_dir / 'original' / 'original_state_dict.pt'
    if args.use_existing_models and original_state_dict_path.exists():
        original_model = build_resnet18(
            num_classes, 
            in_channels,
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
            test_ds=test_ds,
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
        original_state_dict_path = output_dir / 'original' / f'original_cr_state_dict.pt'
        
        print('Adding certified removal mechanism to the original model')
        print('NOTE: THIS WILL NOT PLAY NICE IF OTHER UNLEARNING METHODS ARE USED')
        if args.use_existing_models and original_state_dict_path.exists():
            original_model = build_resnet18(
                num_classes, 
                in_channels,
                pretrained=False, # loading in new weights anyway
                use_differential_privacy=train_params['use_differential_privacy']
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
            eval_model_performance(
                model=original_model, 
                test_ds=test_ds, 
                device=device, 
                batch_size=config_dict['batch_size'], 
                model_dir=output_dir / 'original' / 'cr_performance', 
                num_workers=train_params['num_workers'],
                p_bar_desc=f'Testing original model with CR mechanism'
            )
        print('Certified removal mechanism added successfully')
    
    # Iterate over all the desired forget sets
    score_dicts = {d: {k: [] for k in ('forget_set', 'run_number', 'model', 'score')} for d in distinguisher_params.keys()}
    for forget_set_ind, class_to_forget in enumerate(dataset_params['forget_sets']):
        for run_num in range(config_dict['runs_per_experiment']):
            # We have to reload the original model the unlearning methods appear to act in place
            original_model.load_state_dict(torch.load(str(original_state_dict_path)))
            if isinstance(dataset_params['forget_set_size'], list):
                num_forget = dataset_params['forget_set_size'][forget_set_ind]
            else:
                num_forget = dataset_params['forget_set_size'] if class_to_forget == 'random' else None
            forget_set_name = f'forget_{class_to_forget}{f"_{num_forget}" if num_forget is not None else ""}'
            print(f'Forget Set Size: {forget_set_name}, Run number: {run_num}')
            
            # Create the directory to store everything under
            curr_output_dir = output_dir / forget_set_name / f'run_{run_num}'
            curr_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Construct the forget set
            if args.use_existing_models and (curr_output_dir / 'forget_set.csv').exists():
                forget_set_df = pd.read_csv(str(curr_output_dir / 'forget_set.csv'))
                forget_inds = list(forget_set_df['sample_ind'])
                forget_ds = Subset(full_train_ds, indices=forget_inds)
            else:
                forget_ds, forget_inds = generate_forget_set(
                    full_ds=full_train_ds,
                    num_classes=num_classes,
                    class_to_forget=class_to_forget,
                    num_forget=num_forget,
                    output_dir=curr_output_dir
                )
            
            # Create copies of retain set with and without random augmentations
            retain_inds = [i for i in range(len(full_train_ds)) if i not in forget_inds]
            retain_ds = Subset(full_train_ds, indices=retain_inds)
            retain_ds_random_aug = Subset(full_train_ds_random_aug, indices=retain_inds)
            
            # Train the control model
            if args.use_existing_models and (curr_output_dir / 'control' / 'control_state_dict.pt').exists():
                control_model = build_resnet18(
                    num_classes, 
                    in_channels,
                    pretrained=False, # loading new weights anyway
                    use_differential_privacy=train_params.get('use_differential_privacy', False)
                )
                control_model.load_state_dict(torch.load(str(curr_output_dir / 'control' / 'control_state_dict.pt'), weights_only=True))
                control_model = control_model.to(device)
            else:
                # if train_params['use_certified_removal']:
                #     assert train_params['use_differential_privacy'], 'Certified removal experiments must compare to a DP control model'
                control_model = train_single_model(
                    train_ds=retain_ds_random_aug,
                    test_ds=test_ds,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    batch_size=config_dict['batch_size'],
                    device=device,
                    model_name='control',
                    output_dir=curr_output_dir,
                    **train_params
                )
            
            models_to_score = {'control': control_model.cpu()}
            # Get unlearned models for each unlearning method
            for unlearning_method_key, unlearning_params in unlearning_methods.items():
                unlearning_method = unlearning_method_key.split('.')[0]
                unlearned_model_dir = curr_output_dir / unlearning_method_key
                unlearned_model_ckpt = unlearned_model_dir / f'{unlearning_method}_state_dict.pt'
                print(f'Running {unlearning_method_key}')
                if not args.redo_unlearning and args.use_existing_models and unlearned_model_ckpt.exists():
                    unlearned_model = build_resnet18(
                        num_classes, 
                        in_channels,
                        pretrained=False, # loading new weights anyway
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
                            test_ds=test_ds,
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
                            test_ds=test_ds,
                            num_classes=num_classes,
                            in_channels=in_channels,
                            batch_size=config_dict['batch_size'],
                            device=device,
                            output_dir=unlearned_model_dir,
                            **train_params,
                        )
                
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
                        forget_ds=forget_ds,
                        retain_ds=retain_ds,
                        test_ds=test_ds,
                        batch_size=config_dict['batch_size'],
                        device=device,
                        distinguisher_params=curr_distinguisher_params,
                        num_workers=train_params['num_workers'],
                    )
                    score_dicts[distinguisher]['forget_set'].append(forget_set_name[len("forget_"):])
                    score_dicts[distinguisher]['run_number'].append(run_num)
                    score_dicts[distinguisher]['model'].append(model_name)
                    score_dicts[distinguisher]['score'].append(score)
    
            # Save the resulting scores from each distinguisher into a separate CSV
            for distinguisher, score_dict in score_dicts.items():
                print('Saving out the results')
                score_df = pd.DataFrame(score_dict)
                score_df.to_csv(str(output_dir / f'{distinguisher}_results.csv'), index=False)


if __name__ == "__main__":
    main()
