import argparse
import os
import shutil
import time
import math
import numpy as np
import pickle, copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from helper import accuracy, pgd_attackLinf, gsave, gload, grad_trades
from helper import AverageMeter, find_prob, find_prob2, kl, kl_full, kl2, kl2_full, cross_entropy, cross_entropy_full, criterion, criterion_full
from helper import train_data_c10, train_data_c10_aug2, val_data_c10, train_data_c100, train_data_c100_aug2, val_data_c100, train_data_c10_noaug, Cifar10Subset

from torch.optim.lr_scheduler import MultiStepLR
from WideResNet import WideResNet


parser = argparse.ArgumentParser()
parser.add_argument("name", type=str)
parser.add_argument("loss", type=str, choices=["bce", "ce"])
parser.add_argument("--fd", type=int, default = 40)
parser.add_argument("--sd", type=int, default = 60)
parser.add_argument("--lr", type=float, default=.1)
parser.add_argument("--load", type=bool, default=False)
parser.add_argument("--lam", type=float, default=1.)



args = parser.parse_args()


l_rate = args.lr
momentum = .9
wd = 5e-4
b_s = 128
w_mult= 1
lam = args.lam


PATH = 'models/i/'
PATH_load = 'models/f/35'

#Try soft labels


import wandb
wandb.init(project="BCE_natural", name=args.name)
wandb.config.opt = "SGD"
wandb.config.loss = args.loss
wandb.config.lr = l_rate
wandb.config.momentum = momentum
wandb.config.wd = wd
wandb.config.b_s = b_s
wandb.config.lam = args.lam
wandb.config.load = args.load

num_class = -1

num_class = 10
train_loader = torch.utils.data.DataLoader(Cifar10Subset(10000), batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data_c10, batch_size=128, shuffle=False)

best_test = 0.0

from pathlib import Path
Path(PATH).mkdir(parents=True, exist_ok=True)

def main():
    global best_test
    model = WideResNet(depth=34, num_classes=num_class, widen_factor=w_mult, dropRate=0.0)
    model.cuda()
    if args.load:
        model.load_state_dict(torch.load(PATH_load))

    cudnn.benchmark = True

    opt = optim.SGD(model.parameters(), lr=l_rate, weight_decay=wd, momentum=momentum)

    scheduler = MultiStepLR(opt, milestones=[args.fd, args.sd], gamma=0.1)



    for epoch in range(1, args.sd+21):

        # train for one epoch
        print('lr {:.6f}'.format(opt.param_groups[0]['lr']))

        train(train_loader, model, opt, epoch)
        validate(val_loader, model, epoch)

        wandb.log({'epoch': epoch+1}, step=epoch+1)

        scheduler.step()

def train(train_loader, model, optimizer, epoch):
    """
        Run one train epoch
    """
    losses_n = AverageMeter()
    losses_n_oth = AverageMeter()
    top1_n = AverageMeter()
    top1p_n = AverageMeter()

    model.train()

    for i, (xB, yB) in enumerate(train_loader):
            xB, yB = xB.cuda(), yB.cuda()
            
            yM_n = model(xB)

            g_n = find_prob(yM_n, yB)
            g_n_sec = find_prob2(yM_n, yB)


            if i % 20 == 0:
                print(i)


            loss_n_oth = torch.mean(-1.*torch.log(1.-g_n_sec+1e-3))
            loss_n = criterion(yM_n, yB)
            loss_n_full = criterion_full(yM_n, yB)
            
            if args.loss == 'bce':
                loss_fin = loss_n + lam*loss_n_oth
            if args.loss == 'ce':
                loss_fin = loss_n
            
            with torch.no_grad():
                prec1_n = accuracy(yM_n, yB)
                prec1p_n = torch.mean(g_n)
                losses_n.update(loss_n.item(), yB.size(0))
                losses_n_oth.update(loss_n_oth.item(), yB.size(0))
                top1_n.update(prec1_n.item(), yB.size(0))
                top1p_n.update(prec1p_n.item(), yB.size(0))

        # compute gradient and do SGD step

            optimizer.zero_grad()
            loss_fin.backward()
            optimizer.step()

    wandb.log({'epoch': epoch, 'Train n loss': losses_n.avg, 'Train n error': 100.0-top1_n.avg, 'Train n p error': 100.0-top1p_n.avg,
        'Train n oth loss': losses_n_oth.avg}, step=epoch)
    print('Epoch: [{0}]\t'.format(epoch))

    
def validate(val_loader, model, epoch):
    global best_test
    """
        Run one train epoch
    """
    losses_n = AverageMeter()
    losses_n_oth = AverageMeter()
    top1_n = AverageMeter()
    top1p_n = AverageMeter()

    model.eval()

    for i, (xB, yB) in enumerate(val_loader):
        with torch.no_grad():
            xB, yB = xB.cuda(), yB.cuda()
            
            yM_n = model(xB)

            g_n = find_prob(yM_n, yB)
            g_n_sec = find_prob2(yM_n, yB)


            loss_n_oth = torch.mean(-1.*torch.log(1.-g_n_sec+1e-3))
            loss_n = criterion(yM_n, yB)
            loss_n_full = criterion_full(yM_n, yB)
            
            if args.loss == 'bce':
                loss_fin = loss_n + loss_n_oth
            if args.loss == 'ce':
                loss_fin = loss_n
            
            with torch.no_grad():
                prec1_n = accuracy(yM_n, yB)
                prec1p_n = torch.mean(g_n)
                losses_n.update(loss_n.item(), yB.size(0))
                losses_n_oth.update(loss_n_oth.item(), yB.size(0))
                top1_n.update(prec1_n.item(), yB.size(0))
                top1p_n.update(prec1p_n.item(), yB.size(0))
    
    if best_test < top1_n.avg:
        best_test = top1_n.avg
    wandb.log({'epoch': epoch, 'Test n loss': losses_n.avg, 'Test n error': 100.0-top1_n.avg, 'Test n p error': 100.0-top1p_n.avg,
        'Test n oth loss': losses_n_oth.avg, 'Best Test Error': 100.0 - best_test}, step=epoch)
    print('Epoch: [{0}]\t'.format(epoch))

if __name__ == '__main__':
    main()
