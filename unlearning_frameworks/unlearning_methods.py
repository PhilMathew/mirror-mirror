# Adapted from https://github.com/if-loops/selective-synaptic-dampening/
import random
from typing import *
from copy import deepcopy
import yaml

import torch.utils
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import numpy as np
from torch.nn.utils import clip_grad_norm_
from opacus import PrivacyEngine

from components.certified_removal import *
from .selective_synaptic_dampening.src import ssd
from .certified_deep_unlearning.unlearn import grad_batch, newton_update
import time


def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label in ds:
        classwise_ds[label].append((img, label))
    return classwise_ds


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def training_step(model, batch, device):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    return loss

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y


def UnlearnerLoss(
    output, 
    labels, 
    full_teacher_logits, 
    unlearn_teacher_logits, 
    KL_temperature
):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')


def unlearning_step(
    model,
    unlearning_teacher,
    full_trained_teacher,
    unlearn_data_loader,
    optimizer,
    device,
    KL_temperature,
):
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(
            output=output,
            labels=y,
            full_teacher_logits=full_teacher_logits,
            unlearn_teacher_logits=unlearn_teacher_logits,
            KL_temperature=KL_temperature,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def fit_one_unlearning_cycle(epochs, model, train_loader, lr, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            loss.backward()
            train_losses.append(loss.detach().cpu())

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))


# TODO: Assuming this is bad teacher?
def blindspot_unlearner(
    model,
    unlearning_teacher,
    full_trained_teacher,
    retain_data,
    forget_data,
    epochs=10,
    optimizer="adam",
    lr=0.01,
    batch_size=256,
    device="cuda",
    KL_temperature=1,
):
    # creating the unlearning dataset.
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(
        unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    optimizer = optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        # if optimizer is not a valid string, then assuming it as a function to return optimizer
        optimizer = optimizer  # (model.parameters())

    for epoch in range(epochs):
        loss = unlearning_step(
            model=model.to(device),
            unlearning_teacher=unlearning_teacher.to(device),
            full_trained_teacher=full_trained_teacher.to(device),
            unlearn_data_loader=unlearning_loader,
            optimizer=optimizer,
            device=device,
            KL_temperature=KL_temperature,
        )
    
    return model


# TODO: Originally called blindspot, but afaik it is bad teacher??
def run_bad_teacher(
    model,
    unlearning_teacher,
    retain_ds,
    forget_ds,
    device
):
    student_model = deepcopy(model)
    b_s, KL_temperature = 256, 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    
    retain_train_subset_inds = random.sample(
        range(len(retain_ds)), int(0.3 * len(retain_ds))
    )
    retain_train_subset = Subset(retain_ds, retain_train_subset_inds)

    blindspot_unlearner(
        model=student_model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=model,
        retain_data=retain_train_subset,
        forget_data=forget_ds,
        epochs=1,
        optimizer=optimizer,
        lr=0.0001,
        batch_size=b_s,
        device=device,
        KL_temperature=KL_temperature,
    )

    return student_model


def run_amnesiac(
    model,
    retain_ds,
    forget_ds,
    num_classes,
    device
):
    model = deepcopy(model)
    model = model.to(device)
    
    unlearning_labels = list(range(num_classes))
    unlearning_trainset = []

    for x, clabel in forget_ds:
        rnd = random.choice(unlearning_labels)
        while rnd == clabel:
            rnd = random.choice(unlearning_labels)
        unlearning_trainset.append((x, rnd))

    for x, clabel in retain_ds:
        unlearning_trainset.append((x, clabel))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    fit_one_unlearning_cycle(
        3, model, unlearning_train_set_dl, device=device, lr=0.0001
    )
    
    return model


def run_fisher_forgetting(
    model,
    retain_ds,
    num_classes,
    device
):
    model = deepcopy(model)
    model = model.to(device)
    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = loss_fn(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)

    def get_mean_var(p, is_base_dist=False, alpha=3e-6):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(retain_ds, model)

    fisher_dir = []
    alpha = 1e-6
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
    
    return model

def run_dp_sgd(
    model: nn.Module, 
    forget_ds: Dataset,
    full_train_ds: Dataset,
    device: torch.device,
    eps: float = 1e-3,
    delta: float = 1e-3,
    batch_size: int = 256,
    num_workers: int = 16,
    lr: float = 1e-3,
) -> nn.Module:
    """
    """
    # loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(lr=lr, params = model.parameters())
    # privacy_engine = PrivacyEngine()
    # dataloader = DataLoader(full_train_ds, batch_size=batch_size)
    # model.train()
    # model, optimizer, dataloader = privacy_engine.make_private(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=dataloader,
    #     max_grad_norm=1.0,
    #     noise_multiplier=1.0,
    # )
    
    #     # model.register_forward_hook(forward_hook)
    # # model.register_backward_hook(backward_hook)
    # for epoch in range(50):
    #     for x_batch,y_batch in tqdm(dataloader, total=len(dataloader)):
            
    #         # Run the microbatches
    #         # for idx, (x, y) in enumerate(zip(x_batch, y_batch)):
    #         #     y_hat = model(torch.stack([x]*2).to(device))[0]
    #         y_hat = model(x_batch.to(device))
    #         loss = loss_fn(y_hat, y_batch.to(device))
    #         loss.backward()
    #         # Clip each parameter's per-sample gradient
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         #     for param in model.parameters():
    #         #         per_sample_grad = param.grad.detach().clone()
    #         #         clip_grad_norm_(per_sample_grad, max_norm=1)  # in-place
    #         #         param.accumulated_grads.append(per_sample_grad)  
                
    #         # # Aggregate back
    #         # for param in model.parameters():
    #         #     param.grad = torch.stack(param.accumulated_grads, dim=0).mean(axis=0)

    #         # # Now we are ready to update and add noise!
    #         # for param in model.parameters():
    #         #     param = param - lr * param.grad
    #         #     noise = torch.empty(param.shape).normal_(mean=0, std=eps).to(device)
    #         #     param += noise

    #         #     # param.grad = torch.empty([]).to(device)  # Reset for next iteration
    #         # optimizer.zero_grad()
    return model


def run_certified_removal(
    model: nn.Module, 
    full_train_ds: Dataset,
    forget_inds: List[int],
    device: torch.device,
    lam: float = 1e-6,
    batch_size: int = 256,
    num_workers: int = 16,
) -> nn.Module:
    device = torch.device('cuda')
    # Mostly adapted from https://github.com/facebookresearch/certified-removal/blob/main/test_removal.py
    # Pull out a copy of the fc weights, then remove the fc layer
    w = deepcopy(model.fc.weight.data)
    w = w.T.to(device)
    num_features, num_classes = model.fc.in_features, model.fc.out_features # record these before resetting anything
    model.fc = nn.Identity()
    model = model.to(device)
    
    # Create full data matrix and labels, along with the same things for the retain set
    train_dl = DataLoader(full_train_ds, batch_size=batch_size, num_workers=num_workers)
    X_train, y_train = [], []
    with torch.no_grad():
        for batch in tqdm(train_dl, desc="Generating Data Matrix"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            model_out = model(inputs)
            X_train.append(model_out.detach().cpu())
            y_train.append(labels.detach().cpu())
    X_train, y_train = torch.cat(X_train).view(-1, num_features), onehot(torch.cat(y_train))
    X_train, y_train = X_train.to(device), y_train.to(device)
    
    # Store number of removals
    num_removes = len(forget_inds)

    # initialize K = X^T * X for fast computation of spectral norm
    K = X_train.t().mm(X_train)
    # removal from all one-vs-rest models
    grad_norm_approx = torch.zeros(num_removes).float()
    w_approx = w.clone()
    all_inds = set(range(X_train.shape[0]))
    for i, forget_index in enumerate(tqdm(forget_inds)):
        for k in range(y_train.shape[1]):
            # Create a copy with previously removed samples and the current sample all removed
            s = time.time()
            rem_inds = list(all_inds - set(forget_inds[:(i + 1)]))
            X_rem = X_train[rem_inds, :]
            y_rem = y_train[rem_inds, k]
            # print(f'Part 1 time: {time.time() - s}')
            
            s = time.time()
            # Do all the hessian stuff
            H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, lam, device=device)
            grad_i = lr_grad(w_approx[:, k], X_train[forget_index].unsqueeze(0), y_train[forget_index, k].unsqueeze(0), lam)
            # print(f'Part 2 time: {time.time() - s}')
            
            s = time.time()
            # apply rank-1 down-date to K
            K -= torch.ger(X_train[forget_index], X_train[forget_index])
            spec_norm = spectral_norm(K, device=device)
            # print(f'Part 3 time: {time.time() - s}')
            
            s = time.time()
            Delta = H_inv.mv(grad_i)
            Delta_p = X_train.mv(Delta)
            w_approx[:, k] += Delta
            grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm / 4).cpu()
            # print(f'Part 4 time: {time.time() - s}')
            
    # Update model's classification layer
    model.fc = nn.Linear(num_features, num_classes, bias=False)
    model.fc.weight = nn.Parameter(w_approx.T, requires_grad=True)
    
    return model


def run_ssd(
    model: nn.Module, 
    forget_ds: Dataset,
    full_train_ds: Dataset,
    dampening_constant: float, 
    selection_weighting: float, 
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 16
) -> nn.Module:
    """
    Runs the Selective Synaptic Dampening unlearning method

    :param model: base model to unlearn from
    :type model: nn.Module
    :param forget_ds: forget set as a PyTorch dataset
    :type forget_ds: Dataset
    :param full_train_ds: original training set
    :type full_train_ds: Dataset
    :param dampening_constant: lambda in SSD paper
    :type dampening_constant: float
    :param selection_weighting: alpha in SSD paper
    :type selection_weighting: float
    :param device: device to run stuff on
    :type device: torch.device
    :param batch_size: size of each dataloader batch, defaults to 32
    :type batch_size: int, optional
    :param num_workers: number of workers for dataloader, defaults to 16
    :type num_workers: int, optional
    :return: unlearned model
    :rtype: nn.Module
    """
    parameters = {
        "lower_bound": 1,  # unused
        "exponent": 1,  # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # -1: all layers are available for modification
        "max_layer": -1,  # -1: all layers are available for modification
        "forget_threshold": 1,  # unused
        "dampening_constant": dampening_constant,  # Lambda from paper
        "selection_weighting": selection_weighting,  # Alpha from paper
    }
    model = deepcopy(model)
    
    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters)
    
    model = model.to(device)
    model = model.eval()

    # Calculation of the forget set importances
    forget_dl = DataLoader(forget_ds, batch_size=batch_size, num_workers=num_workers)
    sample_importances = pdr.calc_importance(forget_dl)
    
    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    full_train_dl = DataLoader(full_train_ds, batch_size=batch_size, num_workers=num_workers)
    original_importances = pdr.calc_importance(full_train_dl)

    # Dampen selected parameters
    pdr.modify_weight(original_importances, sample_importances)
    
    return model


def run_certified_deep_unlearning(
    model: nn.Module, 
    retain_ds: Dataset,
    device: torch.device,
    weight_decay: float = 5e-4,
    s1: float = 10,
    s2: float = 1000,
    gamma: float = 1e-2,
    scale: float = 1000,
    std: float = 1e-3,
    batch_size: int = 256,
    num_workers: int = 16,
) -> nn.Module:
    model = model.to(device)
    model.eval()
    
    res_loader = torch.utils.data.DataLoader(retain_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    g = grad_batch(res_loader, weight_decay, model, device)

    delta = newton_update(g, batch_size, retain_ds, weight_decay, gamma, model, s1, s2, scale, device)
    for i, param in enumerate(model.parameters()):
        param.data.add_(-delta[i] + std * torch.randn(param.data.size()).to(device))

    return model


def run_grad_ascent(
    model,
    retain_ds,
    forget_ds,
    num_classes,
    forget_class,
    device,
    batch_size: int = 256,
    num_workers: int = 16
) -> nn.Module:
    retain_dl = DataLoader(retain_ds, batch_size=batch_size, num_workers=num_workers)
    forget_dl = DataLoader(forget_ds, batch_size=batch_size, num_workers=num_workers)
    config = yaml.load('OpenUnlearn/configs/unlearners.yaml')
    
    unlearned_model, _, _ = Unlearner(
        method='GA', 
        model=model, 
        retain_loader=retain_ds, 
        forget_loader=forget_ds, 
        val_loader=None, 
        config=config, 
        device=device
    )
    
    return unlearned_model