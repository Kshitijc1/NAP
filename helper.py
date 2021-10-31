import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset

        
import tensorflow as tf
import pickle

criterion_kl = nn.KLDivLoss(size_average=False)

def gsave(x, gsname):
    with tf.io.gfile.GFile(gsname, "wb") as file:
        pickle.dump(x, file)

def gload(gsname):
    with tf.io.gfile.GFile(gsname, "rb") as file:
        obj = pickle.load(file)
    return obj


def grad(model, inputs, outputs_0, y, dev='cuda', loss_func = nn.CrossEntropyLoss(reduction='sum')):
    y = y.to(dev)
    x = inputs.detach().to(dev).requires_grad_(True)

    
    model.to(dev)
    model.zero_grad()
    output = model(x)

    output = output.requires_grad_(True)

    loss = loss_func(output, y)

    loss.backward()

    g = x.grad.detach()
    return g


def grad_trades(model, inputs, outputs_0, y, dev='cuda', loss_func = nn.CrossEntropyLoss(reduction='sum')):
    y = y.to(dev)
    x = inputs.detach().to(dev).requires_grad_(True)
    criterion_kl = nn.KLDivLoss()
    model.to(dev)
    model.zero_grad()
    output = model(x)

    output = output.requires_grad_(True)

    loss = criterion_kl(F.log_softmax(output, dim=1),
                                       F.softmax(outputs_0, dim=1))

    loss.backward()

    g = x.grad.detach()
    return g

def pgd_attackLinf(targ_model, X, Y, targetted=False, nSteps=10, eps=8.0/255.0, rand = True, alpha = 2.0/255.0, grad_func=grad):
    X = X.cuda()
    Y = Y.cuda()
    targ_model.to('cuda')

    if rand:
        delta = torch.randn_like(X).cuda()
        delta = (eps/2.)*torch.sign(delta)
    else:
        delta = torch.zeros_like(X).cuda().detach()
    
    if grad_func == grad_trades:
        delta = 0.001*torch.randn(X.shape).cuda().detach()
    
    with torch.no_grad():
        outputs_0 = targ_model(X).cuda().detach()
    for ii in range(nSteps):

        g = grad_func(targ_model, X + delta, outputs_0, Y)

        with torch.no_grad():
            delta += alpha * torch.sign(g)
            delta = torch.clamp(delta, min=-eps, max=eps)
            delta = (torch.clamp(X + delta, min=0, max=1) - X).detach() # valid image box

    xAdv = X + delta
    return xAdv.detach()




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]

transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

transform_train_aug2 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])

transform_train_aug3 = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])

transform_train_aug4 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.ToTensor(),
        ])

transform_train_aug5 = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.ToTensor(),
        ])



# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor()
    #normalize,
])


train_data_c10 = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
train_data_c10_aug2 = datasets.CIFAR10(root='./data', train=True, transform=transform_train_aug2, download=True)
train_data_c10_noaug = datasets.CIFAR10(root='./data', train=True, transform=transform_test, download=True)

val_data_c10 = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

train_data_c100 = datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
train_data_c100_aug2 = datasets.CIFAR100(root='./data', train=True, transform=transform_train_aug2, download=True)
train_data_c100_aug3 = datasets.CIFAR100(root='./data', train=True, transform=transform_train_aug3, download=True)
train_data_c100_aug4 = datasets.CIFAR100(root='./data', train=True, transform=transform_train_aug4, download=True)
train_data_c100_aug5 = datasets.CIFAR100(root='./data', train=True, transform=transform_train_aug5, download=True)


val_data_c100 = datasets.CIFAR100(root='./data', train=False, transform=transform_test, download=True)

val_data_svhn = datasets.SVHN(root='./data', split='test', transform=transform_test, download=True)

train_data_svhn = datasets.SVHN(root='./data', split='train', transform=transform_test, download=True)

def find_prob(yM, yB):
    y = nn.functional.one_hot(yB, yM.size(1)).double()
    y2 = y.ge(.5)
    g = torch.masked_select(nn.functional.softmax(yM, dim=1), y2)
    g = torch.reshape(g, [yM.size(0)])
    return g

def find_prob2(yM, yB):
    y = nn.functional.one_hot(yB, yM.size(1)).double()
    y1 = y.le(.5)
    g_oth = torch.masked_select(nn.functional.softmax(yM, dim=1), y1)
    g_oth = torch.reshape(g_oth, [yB.size(0), yM.size(1)-1])
    g_sec = torch.max(g_oth, dim=1)[0]
    return g_sec


criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
criterion_full = nn.CrossEntropyLoss(reduction='none').cuda()
def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def cross_entropy_full(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.sum(- soft_targets * logsoftmax(pred), 1)

def kl_full(yM_n, yM_a):
    return cross_entropy_full(yM_a, nn.functional.softmax(yM_n, dim=1))-cross_entropy_full(yM_n, nn.functional.softmax(yM_n, dim=1))

def kl(yM_n, yM_a):
    return cross_entropy(yM_a, nn.functional.softmax(yM_n, dim=1))-cross_entropy(yM_n, nn.functional.softmax(yM_n, dim=1))
def kl2(g_n, g_a):
    return torch.mean(g_n*(torch.log(g_n+1e-3)-torch.log(g_a+1e-3))+(1-g_n)*(torch.log(1-g_n+1e-3)-torch.log(1-g_a+1e-3)))

def kl2_full(g_n, g_a):
    return g_n*(torch.log(g_n+1e-3)-torch.log(g_a+1e-3))+(1-g_n)*(torch.log(1-g_n+1e-3)-torch.log(1-g_a+1e-3))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Cifar10Subset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, sz):
        self.sz = sz
        self.data = datasets.CIFAR10(root='./data', train=True, transform=transform_test, download=True)

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)