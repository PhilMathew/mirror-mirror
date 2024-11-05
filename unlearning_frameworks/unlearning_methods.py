# Adapted from https://github.com/if-loops/selective-synaptic-dampening/
import random
from typing import *
from copy import deepcopy

from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import numpy as np

from .selective_synaptic_dampening.src import ssd


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
    forget_class,
    device
):
    model = deepcopy(model)
    model = model.to(device)
    
    unlearning_labels = list(range(num_classes))
    unlearning_trainset = []

    unlearning_labels.remove(forget_class)

    for x, clabel in forget_ds:
        unlearning_trainset.append((x, random.choice(unlearning_labels)))

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


# # TODO: Actually read the paper to understand what's happening in this unlearning method
# def run_UNSIR(
#     model,
#     retain_ds,
#     forget_ds,
#     num_classes,
#     forget_class,
#     device,
# ):
#     classwise_train = get_classwise_ds(
#         ConcatDataset((retain_ds, forget_ds)),
#         num_classes,
#     )
#     noise_batch_size = 32
#     # collect some samples from each class
#     num_samples = 500
#     retain_samples = []
#     for i in range(num_classes):
#         if i != forget_class:
#             retain_samples += classwise_train[i][:num_samples]

#     forget_class_label = forget_class
#     img_shape = next(iter(retain_ds.dataset))[0].shape[-1]
#     noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
#     noise = UNSIR_noise_train(
#         noise, model, forget_class_label, 250, noise_batch_size, device=device
#     )
#     noisy_loader = UNSIR_create_noisy_loader(
#         noise, forget_class_label, retain_samples, noise_batch_size, device=device
#     )
#     # impair step
#     _ = fit_one_unlearning_cycle(
#         1, model, noisy_loader, device=device, lr=0.0001
#     )
#     # repair step
#     other_samples = []
#     for i in range(len(retain_samples)):
#         other_samples.append(
#             (
#                 retain_samples[i][0].cpu(),
#                 torch.tensor(retain_samples[i][2]),
#                 torch.tensor(retain_samples[i][2]),
#             )
#         )

#     heal_loader = torch.utils.data.DataLoader(
#         other_samples, batch_size=128, shuffle=True
#     )
#     _ = fit_one_unlearning_cycle(
#         1, model, heal_loader, device=device, lr=0.0001
#     )

#     return model
