from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager



# From https://github.com/if-loops/selective-synaptic-dampening.git
class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        """
        Warmup training learning rate scheduler
    
        :param optimizer: optimizer (e.g. SGD)
        :param total_iters: Number of iterations of warmup phase
        """
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # We will use the first m batches, and set the learning rate to base_lr * m / total_iters
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


def _train_step(
    model: nn.Module,
    train_dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_differential_privacy: bool,
    history: Dict[str, List[float]],
    epoch: int,
    privacy_engine: Optional[PrivacyEngine] = None,
    target_delta: Optional[float] = None,
):
    p_bar = tqdm(train_dl, desc=f'Epoch {epoch}')
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
        if (i+1) % 200 == 0:
            if use_differential_privacy:
                try:
                    epsilon = privacy_engine.get_epsilon(target_delta)
                except ValueError as e:
                    epsilon = np.nan
                postfix_str = f'Train Loss: {train_loss / (i + 1):.4f}, Train Accuracy: {100 * train_acc / (i + 1):.4f}%, epsilon: {epsilon:.4f}, delta: {target_delta:.4f}'
            else:
                postfix_str = f'Train Loss: {train_loss / (i + 1):.4f}, Train Accuracy: {100 * train_acc / (i + 1):.4f}%'
            p_bar.set_postfix_str(postfix_str)
            p_bar.update()

    postfix_str = f'After Epoch {epoch} Train Loss: {train_loss / (i + 1):.4f}, Train Accuracy: {100 * train_acc / (i + 1):.4f}%, epsilon: {epsilon:.4f}, delta: {target_delta:.4f}'
    print(postfix_str)
    history['train_loss'].append(train_loss / (len(train_dl)))
    history['train_acc'].append(train_acc / (len(train_dl)))
    

def train_model(
    model: nn.Module, 
    train_ds: Dataset, 
    device: torch.device,
    val_ds: Optional[Dataset] = None,
    batch_size: int = 32, 
    num_epochs: int = 10,
    lr: int = 1e-3,
    num_workers: int = 16,
    warmup_epochs: int = 1,
    use_differential_privacy: bool = False,
    max_physical_batch_size: int = 128,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Trains a PyTorch model on a given dataset.

    :param model: Model to train
    :type model: nn.Module
    :param train_ds: Training dataset
    :type train_ds: Dataset
    :param device: Device to train on
    :type device: torch.device
    :param val_ds: Validation dataset, defaults to None
    :type val_ds: Optional[Dataset], optional
    :param batch_size: Number of examples in each batch, defaults to 32
    :type batch_size: int, optional
    :param num_epochs: Number of training epochs, defaults to 10
    :type num_epochs: int, optional
    :param lr: Initial learning rate, defaults to 1e-3
    :type lr: int, optional
    :param num_workers: Number of workers for dataloader, defaults to 16
    :type num_workers: int, optional
    :param warmup_epochs: Number of warmup epochs, defaults to 1
    :type warmup_epochs: int, optional
    :return: Dictionary of training history, including training and validation loss and accuracy
    :rtype: Dict[str, List[float]]
    """
    # Model-related things
    model = model.to(device)
    model.train()
    print(f"Num Workers: {num_workers}")
    # Data-related stuff
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    
    # Optimizer, loss function, and LR scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = nn.CrossEntropyLoss()
    if use_differential_privacy:
        if 'epsilon' in kwargs and 'delta' in kwargs:
            target_delta = kwargs['delta']
            target_epsilon = kwargs['epsilon']
        else:
            assert 'lambda' in kwargs.keys(), "Either just lambda or both epsilon and delta must be specified for DP-SGD"
            
            # Based on theoretical values in paper
            target_delta = 2**(-kwargs['lambda']) # just needs to be negligible in lambda
            target_epsilon = np.log(1 + target_delta)
        
        privacy_engine = PrivacyEngine()
        model, optimizer, train_dl = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dl,
            target_delta=target_delta,
            target_epsilon=target_epsilon,
            epochs=num_epochs,
            max_grad_norm=1.2
        )
        # print(f'Using sigma = {optimizer.noise_multiplier}')
    
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc')}
    for epoch in range(1, num_epochs + 1):

        if use_differential_privacy:
            print(f"Max Physical Batch Size: {max_physical_batch_size}, Num WOrkers: {num_workers}")
            with BatchMemoryManager(data_loader=train_dl, max_physical_batch_size=max_physical_batch_size, optimizer=optimizer) as train_dl_mem:
                _train_step(
                    model=model,
                    train_dl=train_dl_mem,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    use_differential_privacy=use_differential_privacy,
                    history=history,
                    epoch=epoch,
                    privacy_engine=privacy_engine,
                    target_delta=target_delta
                )
        else:
            _train_step(
                model=model,
                train_dl=train_dl,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                use_differential_privacy=use_differential_privacy,
                history=history,
                epoch=epoch,
            )
        
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
        
        scheduler.step() # update learning rate
    
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
    """
    Tests a PyTorch model on a given dataset.

    :param model: Model to test
    :type model: nn.Module
    :param test_ds: Test dataset
    :type test_ds: Dataset
    :param device: Device to test on
    :type device: torch.device
    :param batch_size: Number of examples in each batch, defaults to 32
    :type batch_size: int, optional
    :param num_workers: Number of workers for dataloader, defaults to 16
    :type num_workers: int, optional
    :param p_bar_desc: Progress bar description for logging, defaults to 'Testing Model'
    :type p_bar_desc: str, optional
    :param return_preds_and_labels: Whether to return predictions and labels, defaults to False
    :type return_preds_and_labels: bool, optional
    :return: Either loss and accuracy, or loss, accuracy, predictions, and labels if return_preds_and_labels is True
    :rtype: Union[Tuple[float, float, torch.Tensor, torch.Tensor], Tuple[float, float]]
    """
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

