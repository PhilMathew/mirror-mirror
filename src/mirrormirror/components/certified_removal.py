import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def onehot(y):
    y_onehot = -torch.ones(y.size(0), (y.max() + 1).int().item()).float()
    y_onehot.scatter_(1, y.long().unsqueeze(1), 1)
    return y_onehot


def lr_loss(w, X, y, lam):
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2


def lr_eval(w, X, y):
    return X.mv(w).sign().eq(y).float().mean()


def lr_grad(w, X, y, lam):
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z-1) * y) + lam * X.size(0) * w


def lr_hessian_inv(w, X, y, lam, device, batch_size=50000):
    z = torch.sigmoid(X.mv(w).mul_(y))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i+1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()


def lr_optimize(X, y, lam, device, b=None, num_steps=100, tol=1e-10, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)
    optimizer = torch.optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-20)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data


def ovr_lr_loss(w, X, y, lam, device, weight=None):
    z = batch_multiply(X, w, device=device).mul_(y)
    if weight is None:
        return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2
    else:
        return -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2


def ovr_lr_optimize(X, y, lam, device, weight=None, b=None, num_steps=100, tol=1e-10, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1), y.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return ovr_lr_loss(w, X, y, lam, weight=weight, device=device)
        else:
            return ovr_lr_loss(w, X, y, lam, weight=weight, device=device) + (b * w).sum() / X.size(0)
    optimizer = torch.optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-10)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = ovr_lr_loss(w, X, y, lam, weight=weight, device=device)
        if b is not None:
            if weight is None:
                loss += (b * w).sum() / X.size(0)
            else:
                loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
        loss.backward()
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data


def batch_multiply(A, B, device, batch_size=500000):
    if A.is_cuda:
        if len(B.size()) == 1:
            return A.mv(B)
        else:
            return A.mm(B)
    else:
        out = []
        num_batch = int(math.ceil(A.size(0) / float(batch_size)))
        with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i+1) * batch_size, A.size(0))
                A_sub = A[lower:upper]
                A_sub = A_sub.to(device)
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
        return torch.cat(out, dim=0).to(device)


def spectral_norm(A, device, num_iters=20):
    x = torch.randn(A.size(0)).float().to(device)
    norm = 1
    for i in range(num_iters):
        x = A.mv(x)
        norm = x.norm()
        x /= norm
    return math.sqrt(norm)  


class L2NormLayer(nn.Module):
    def __init__(self) -> None:
        super(L2NormLayer, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)
