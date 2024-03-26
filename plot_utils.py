from typing import *
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_history(train_hist: Dict[str, Sequence[float]]) -> plt.Figure:
    plot_val = 'val_loss' in train_hist.keys()
    if plot_val:
        fig, (train_loss_ax, train_acc_ax) = plt.subplots(1, 2, figsize=(20, 10))
    else:
        fig, ((train_loss_ax, train_acc_ax), (val_loss_ax, val_acc_ax)) = plt.subplots(2, 2, figsize=(20, 10))
        
        val_loss_ax.plot(train_hist['val_loss'])
        val_loss_ax.set(title='Validation Loss', xlabel='Epoch', ylabel='Loss')
        
        val_acc_ax.plot(train_hist['val_acc'])
        val_acc_ax.set(title='Validation Accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=[0, 1])
    
    train_loss_ax.plot(train_hist['train_loss'])
    train_loss_ax.set(title='Train Loss', xlabel='Epoch', ylabel='Loss')
    
    train_acc_ax.plot(train_hist['train_acc'])
    train_acc_ax.set(title='Train Accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=[0, 1])

    return fig


def plot_confmat(cm, save_path='confmat.png', title='', label_mapping: dict = None):
    fig, ax = plt.subplots(1, 1, num=2, figsize=(15, 10))
    
    cm = np.array(cm)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize the confusion matrix
    cm_norm[np.isnan(cm_norm)] = 0

    annot = np.zeros_like(cm, dtype=object)
    for i in range(annot.shape[0]):  # Creates an annotation array for the heatmap
        for j in range(annot.shape[1]):
            annot[i][j] = f'{cm[i][j]}\n{round(cm_norm[i][j] * 100, ndigits=3)}%'
    
    ax = sns.heatmap(cm_norm, annot=annot, fmt='', cbar=True, cmap=plt.cm.magma, vmin=0, ax=ax) # plot the confusion matrix
    ax.set(xlabel='Predicted Label', ylabel='Actual Label', title=f'{title} (CM Trace: {cm_norm.trace():.4f})')
    
    if label_mapping:
        ticks = sorted(label_mapping.keys(), key=(lambda x: label_mapping[x]))
        ax.set(xticklabels=ticks, yticklabels=ticks)

    fig.savefig(str(save_path))
