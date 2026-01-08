# Adapted from https://github.com/if-loops/selective-synaptic-dampening/
import random
from typing import *
from copy import deepcopy
from dataclasses import dataclass
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
from unlearning_frameworks.selective_synaptic_dampening.src import ssd
from unlearning_frameworks.certified_deep_unlearning.unlearn import grad_batch, newton_update
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
    forget_ds,
    device,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    epochs: int = 5,
    batch_size: int = 256,
    num_workers: int = 16
) -> nn.Module:
    forget_dl = DataLoader(forget_ds, batch_size=batch_size, num_workers=0)
    print('NUM_WORKERS IS DISCARDED TO AVOID SLOWDOWNS, CHANGE THE CODE TO USE A VALID NUMBER OF WORKERS IF NECESSARY.')
    
    unlearned_model = deepcopy(model)
    unlearned_model.to(device)
    
    optimizer = torch.optim.SGD(unlearned_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    loss_func = nn.CrossEntropyLoss().to(device)

    for epoch in tqdm(range(1, epochs + 1), desc= "Gradient ascent unlearning"):
        loss_list = []
        for imgs, labels in forget_dl:
            imgs, labels = imgs.to(device), labels.long().to(device)
            unlearned_model.zero_grad()
            output = unlearned_model(imgs)
            # gradient ascent loss
            loss = (-1) * loss_func(output, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

    return unlearned_model


# Adapted from https://github.com/MartinPawelczyk/OpenUnlearn.git
def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def sgda_adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    new_lr = opt.sgda_learning_rate
    if steps > 0:
        new_lr = opt.sgda_learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr

def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def l2_difference(model1, model2):
    l2_diff = 0.0
    # Ensure both models are in the same state (e.g., both in eval mode)
    model1.eval()
    model2.eval()

    with torch.no_grad():
        for (param1, param2) in zip(model1.parameters(), model2.parameters()):
            # Check if both parameters are on the same device and are of the same shape
            if param1.device != param2.device or param1.shape != param2.shape:
                raise ValueError("Models have parameters on different devices or with different shapes")
            
            # Compute the squared L2 norm of the difference between the parameters
            param_diff = param1 - param2
            l2_diff += torch.norm(param_diff, p=2).item()**2

    # Return the square root of the sum of squared differences
    return l2_diff**0.5

def compute_classification_loss(pred_scores: torch.tensor, labels: torch.tensor):
    loss_fct = nn.CrossEntropyLoss()
    lm_loss = loss_fct(pred_scores, labels.view(-1))
    return lm_loss


def train_distill(epoch, train_loader, module_list, swa_model, 
                  criterion_list, optimizer, opt, split, quiet=False):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    # end = time.time()
    for idx, data in enumerate(train_loader):
        if isinstance(data, dict):
            inputs = data['input_embeds'].to(opt.DEVICE)
            attention_mask = data['attention_mask'].to(opt.DEVICE)
            targets = data['label'].to(opt.DEVICE)
        else:
            inputs, targets = data
            inputs, targets = inputs.to(opt.DEVICE), targets.to(opt.DEVICE)
        # data_time.update(time.time() - end)
        # inputs = inputs.float()
        # ===================forward=====================
        model_s = model_s.to(opt.DEVICE) 
        model_t = model_t.to(opt.DEVICE) 
        swa_model = swa_model.to(opt.DEVICE) 
        if isinstance(data, dict):
            logit_s = model_s(inputs_embeds=inputs, attention_mask=attention_mask).logits
        else:
            inputs = inputs.float()
            logit_s = model_s(inputs)
        with torch.no_grad():
            if isinstance(data, dict):
                logit_t = model_t(inputs_embeds=inputs, attention_mask=attention_mask).logits
            else:
                logit_t = model_t(inputs)
        # cls + kl div
        loss_cls = criterion_cls(logit_s, targets)
        loss_div = criterion_div(logit_s, logit_t)
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        else:
            raise NotImplementedError(opt.distill)
        if split == "minimize":
            loss = (opt.gamma * loss_cls) + (opt.alpha * loss_div) + (opt.beta * loss_kd)
        elif split == "maximize":
            loss = -loss_div
        loss = loss + param_dist(model_s, swa_model, opt.smoothing)
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================meters=====================
        # batch_time.update(time.time() - end)
        # end = time.time()
    if split == "minimize":
        return top1.avg, losses.avg
    else:
        return kd_losses.avg

def scrub_loop(args,
               optimizer,
               model,
               criterion_cls,
               module_list,
               swa_model,
               criterion_list,
               retain_loader,
               forget_loader,
               verbose=False):
    maximize_loss = None

    # print(f"total epochs : {args.sgda_epochs}")
    for epoch in tqdm(range(1, args.sgda_epochs + 1), desc='Running SCRUB'):
        # print(f"Epoch {epoch} ...")
        lr = sgda_adjust_learning_rate(epoch, args, optimizer)
        if verbose:
            print("==> scrub unlearning ...")
            print(f"validating - ")

        maximize_loss = 0
        if epoch <= args.msteps:
            if verbose:
                print(f"train distill 1")
            # maximize loss on the forget set
            maximize_loss = train_distill(epoch, forget_loader, module_list,
                                          swa_model, criterion_list, optimizer,
                                          args, 
                                          "maximize")
        if verbose:
            print(f"train distill 2 :")
        # minimize loss on the retain set
        train_acc, train_loss = train_distill(epoch, retain_loader,
                                              module_list, swa_model,
                                              criterion_list, optimizer, args,
                                              "minimize")

        if epoch >= args.sstart:
            # print("update params")
            swa_model.update_parameters(model)

        if verbose:
            print(
                "maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}"
                .format(maximize_loss, train_loss, train_acc))
        student = module_list[0]
        teacher = module_list[-1]
        # print(f"{epoch} -- difference between teacher and student - {l2_difference(student, teacher)}")

    return model

@dataclass
class SCRUB_args:
    alpha: float
    beta: float
    epochs: int
    batch_size: int
    lr: float
    DEVICE: str
    
    noise_var: float = 0.000
    lr_scheduler = None
    gamma: float = 0.99
    smoothing: float = 0.0
    clip: float = 0.2
    sstart: float = 10
    kd_T: float = 4
    distill: str = 'kd'
    
    lr_decay_epochs: tuple[int] = (3, 5, 9)
    lr_decay_rate: float = 0.1
    sgda_weight_decay: float = 5e-4
    sgda_momentum: float = 0.9
    
    def __post_init__(self):
        self.sgda_batch_size: int = self.batch_size
        self.del_batch_size: int = self.batch_size
        self.msteps: int = self.epochs
        self.sgda_epochs: int = self.epochs
        self.sgda_learning_rate: float = self.lr # 0.0005


# SCRUB: https://github.com/meghdadk/SCRUB/tree/main
def run_scrub(
    model,
    forget_ds,
    retain_ds,
    alpha,
    beta,
    lr,
    epochs,
    momentum,
    weight_decay,
    batch_size,
    num_workers,
    device,
):
    orig_model = deepcopy(model)
    unlearned_model = deepcopy(model)
    forget_dl = DataLoader(forget_ds, batch_size=batch_size, num_workers=0)
    retain_dl = DataLoader(retain_ds, batch_size=batch_size, num_workers=0)
    print('NUM_WORKERS IS DISCARDED TO AVOID SLOWDOWNS, CHANGE THE CODE TO USE A VALID NUMBER OF WORKERS IF NECESSARY.')
    
    module_list = nn.ModuleList([])
    module_list.append(unlearned_model)  # student.
    module_list.append(orig_model)   # teacher.
    trainable_list = nn.ModuleList([])
    trainable_list.append(unlearned_model)   
    
    args = SCRUB_args( # literally just so the existing SCRUB functions work
        alpha=alpha,
        beta=beta,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        DEVICE=device,
    )
    
    optimizer = torch.optim.SGD(
        trainable_list.parameters(),
        lr=args.sgda_learning_rate,
        momentum=args.sgda_momentum,
        weight_decay=args.sgda_weight_decay
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)   # other knowledge distillation loss

    def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        return ((1 - beta) * averaged_model_parameter) + (beta * model_parameter)

    # continue training using SCRUB on Retain set + Forget set
    swa_model = torch.optim.swa_utils.AveragedModel(unlearned_model, avg_fn=avg_fn)
    unlearned_model = scrub_loop(
        args, 
        optimizer, 
        unlearned_model,
        criterion_cls, 
        module_list, 
        swa_model,
        criterion_list, 
        retain_dl, 
        forget_dl
    )
    
    return unlearned_model