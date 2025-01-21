from typing import *
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_history(train_hist: Dict[str, Sequence[float]], save_path='train_hist.png'):
    """
    Plots the training history of a model.

    :param train_hist: Dictionary of training history
    :type train_hist: Dict[str, Sequence[float]]
    :param save_path: Path to save the plot, defaults to 'train_hist.png'
    :type save_path: str, optional
    """
    plot_val = 'val_loss' in train_hist.keys()
    
    if plot_val:
        fig, ((train_loss_ax, train_acc_ax), (val_loss_ax, val_acc_ax)) = plt.subplots(2, 2, figsize=(20, 10))
        
        val_loss_ax.plot(train_hist['val_loss'])
        val_loss_ax.set(title='Validation Loss', xlabel='Epoch', ylabel='Loss')
        
        val_acc_ax.plot(train_hist['val_acc'])
        val_acc_ax.set(title='Validation Accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=[0, 1])
    else:
        fig, (train_loss_ax, train_acc_ax) = plt.subplots(1, 2, figsize=(20, 10))
    
    train_loss_ax.plot(train_hist['train_loss'])
    train_loss_ax.set(title='Train Loss', xlabel='Epoch', ylabel='Loss')
    
    train_acc_ax.plot(train_hist['train_acc'])
    train_acc_ax.set(title='Train Accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=[0, 1])

    fig.savefig(str(save_path))


def plot_confmat(cm: Union[List, np.ndarray], save_path='confmat.png', title='', label_mapping: dict = None):
    """
    Plots a given confusion matrix

    :param cm: Confusion matrix to plot
    :type cm: Union[List, np.ndarray]
    :param save_path: Path to save the plot, defaults to 'confmat.png'
    :type save_path: str, optional
    :param title: Plot title, defaults to ''
    :type title: str, optional
    :param label_mapping: Dictionary mapping labels to their names, e.g. {0: 'cat', 1: 'dog'}, defaults to None
    :type label_mapping: dict, optional
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    cm = np.array(cm)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize the confusion matrix
    cm_norm[np.isnan(cm_norm)] = 0

    annot = np.zeros_like(cm, dtype=object)
    for i in range(annot.shape[0]):  # Creates an annotation array for the heatmap
        for j in range(annot.shape[1]):
            annot[i][j] = f'{cm[i][j]}\n{round(cm_norm[i][j] * 100, ndigits=3)}%'
    
    ax = sns.heatmap(cm_norm, annot=annot, fmt='', cbar=True, cmap=plt.cm.magma, vmin=0, vmax=1, ax=ax) # plot the confusion matrix
    ax.set(xlabel='Predicted Label', ylabel='Actual Label', title=f'{title} (CM Trace: {cm_norm.trace():.4f})')
    
    if label_mapping:
        ticks = sorted(label_mapping.keys(), key=(lambda x: label_mapping[x]))
        ax.set(xticklabels=ticks, yticklabels=ticks)
    
    # fig.tight_layout()
    fig.savefig(str(save_path))
