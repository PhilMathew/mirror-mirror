from argparse import ArgumentParser
import json
from tqdm import tqdm
from pathlib import Path
from torchvision.models.resnet import Bottleneck
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
from typing import *
import pandas as pd
from sklearn.metrics import confusion_matrix

from components.resnet import ResNet, build_resnet50
from utils.plot_utils import plot_history, plot_confmat
from components.datasets import init_full_ds





def train_model(
    model: nn.Module, 
    train_ds: Dataset, 
    device: torch.device,
    val_ds: Optional[Dataset] = None,
    batch_size: int = 32, 
    num_epochs: int = 10,
    lr: int = 1e-3,
    num_workers: int = 16
) -> Dict[str, List[float]]:
    # Model-related things
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Data-related stuff
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc')}
    for epoch in range(num_epochs):
        p_bar = tqdm(train_dl, desc=f'Epoch {epoch + 1}')
        train_loss, train_acc = 0, 0
        model.train() # in case an eval loop set it to evaluation
        for i, batch in enumerate(p_bar):
            optimizer.zero_grad()
            
            # Sorting out data
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Run examples through model
            preds = model(inputs)
            
            # Backprop
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics calculation
            train_loss += loss.item()
            train_acc += (torch.sum(F.softmax(preds, dim=-1).argmax(dim=-1) == labels) / labels.shape[0]).item()
            
            p_bar.set_postfix_str(
                f'Train Loss: {train_loss / (i + 1):.4f}, Train Accuracy: {100 * train_acc / (i + 1):.4f}%'
            )
            p_bar.update()
        history['train_loss'].append(train_loss / (len(train_dl)))
        history['train_acc'].append(train_acc / (len(train_dl)))
        
        if val_ds is not None:
            val_loss, val_acc = test_model(
                model,
                val_ds, 
                batch_size=batch_size,
                device=device,
                p_bar_desc='Running Validation',
                return_preds_and_labels=False
            )
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {100 * val_acc:.4f}%')
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
    
    # Remove empty keys from history
    history = {k: v for k, v in history.items() if len(v) > 0}
    
    return history


def test_model(
    model: nn.Module, 
    test_ds: Dataset, 
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 16, 
    p_bar_desc: str = 'Testing Model',
    return_preds_and_labels : bool = False
) -> Union[Tuple[float, float, torch.Tensor, torch.Tensor], Tuple[float, float]]:
    model = model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    
    with torch.no_grad():
        test_loss = 0
        all_preds, all_labels = [], []
        for batch in tqdm(test_dl, desc=p_bar_desc):
            
            # Sorting out data
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Run examples through model
            preds = model(inputs)
            
            # Metrics calculation
            loss = loss_fn(preds, labels)
            test_loss += loss.item()
            
            # Update running collections of things
            preds = F.softmax(preds, dim=-1).argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        test_loss /= len(test_dl)
        test_acc = torch.sum(all_preds == all_labels) / len(test_ds)

    return (test_loss, test_acc, all_preds, all_labels) if return_preds_and_labels else (test_loss, test_acc)


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
    full_train_ds, test_ds = init_full_ds(args.ds_type)
    match args.ds_type:
        case 'MNIST':
            num_classes = 10
            in_channels = 1
        case 'CIFAR10':
            num_classes = 10
            in_channels = 3
        case 'CIFAR100':
            num_classes = 100
            in_channels = 3
        case _:
            raise ValueError(f'{args.ds_type} is an invalid dataset type')
    
    # Create forget and retain sets
    forget_inds = list(forget_set_df['sample_ind'])
    forget_ds = Subset(full_train_ds, indices=forget_inds)
    retain_inds = [i for i in range(len(full_train_ds)) if i not in forget_inds]
    retain_ds = Subset(full_train_ds, indices=retain_inds)
    
    # Initialize and train M1 (full dataset) and M3 (only the retain set)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m1, m3 = build_resnet50(num_classes, in_channels), build_resnet50(num_classes, in_channels)
    print('Training M1')
    m1_hist = train_model(
        model=m1,
        train_ds=full_train_ds,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.learning_rate
    )
    print('Training M3')
    m3_hist = train_model(
        model=m3,
        train_ds=retain_ds,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.learning_rate
    )
    
    # Save out M1-related stuff
    m1_dir = output_dir / 'm1'
    m1_dir.mkdir(exist_ok=True)
    torch.save(m1.state_dict(), str(m1_dir / 'm1_state_dict.pt')) # model
    with open(str(m1_dir / 'm1_train_hist.json'), 'w') as f: # train history
        json.dump(m1_hist, f, indent=4)
    if args.num_epochs > 1: # plots are useless for training in a single epoch
        plot_history(m1_hist, str(m1_dir / 'm1_train_hist.png')) # train history plots
    
    # Save out M3-related stuff
    m3_dir = output_dir / 'm3'
    m3_dir.mkdir(exist_ok=True)
    torch.save(m3.state_dict(), str(m3_dir / 'm3_state_dict.pt')) # model
    with open(str(m3_dir / 'm3_train_hist.json'), 'w') as f: # train history
        json.dump(m3_hist, f, indent=4)
    if args.num_epochs > 1: # plots are useless for training in a single epoch
        plot_history(m3_hist, str(m3_dir / 'm3_train_hist.png')) # train history plots
    
    # Save confmats from testing
    m1_test_loss, m1_test_acc, m1_preds, m1_labels = test_model(
        m1, 
        test_ds,
        device, 
        batch_size=args.batch_size,
        return_preds_and_labels=True, 
        p_bar_desc='Testing M1'
    )
    m3_test_loss, m3_test_acc, m3_preds, m3_labels = test_model(
        m3, 
        test_ds, 
        device, 
        batch_size=args.batch_size, 
        return_preds_and_labels=True, 
        p_bar_desc='Testing M3'
    )
    m1_confmat, m3_confmat = confusion_matrix(m1_labels, m1_preds), confusion_matrix(m3_labels, m3_preds)
    plot_confmat(m1_confmat, save_path=str(m1_dir / 'm1_confmat.png'), title=f'Loss: {m1_test_loss:.4f}, Accuracy: {100 * m1_test_acc:.4f}%')
    plot_confmat(m3_confmat, save_path=str(m3_dir / 'm3_confmat.png'), title=f'Loss: {m3_test_loss:.4f}, Accuracy: {100 * m3_test_acc:.4f}%')


if __name__ == '__main__':
    main()
