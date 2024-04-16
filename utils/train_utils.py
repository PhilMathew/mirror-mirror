from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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

