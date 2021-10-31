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

from advertorch.attacks import LinfSPSAAttack

from helper import accuracy, pgd_attackLinf, gsave, gload, grad_trades
from helper import AverageMeter, find_prob, find_prob2, kl, kl_full, kl2, kl2_full, cross_entropy, cross_entropy_full, criterion, criterion_full
from helper import val_data_c10

from torch.optim.lr_scheduler import MultiStepLR
from WideResNet import WideResNet

import wandb 



parser = argparse.ArgumentParser()
parser.add_argument("name", type=str)
parser.add_argument("load", type=str)
parser.add_argument("w", type=int, default=4)
parser.add_argument("--b_s", type=int, default=25)
parser.add_argument("--iter", type=int, default=100)
parser.add_argument("--spsa_b_s", type=int, default=8192)

args = parser.parse_args()

wandb.init(project="spsa", name=args.name)
wandb.config.update(args, allow_val_change=True)




b_s = args.b_s
num_class = 10
w_mult = args.w



def main():
    global best_test
    model = WideResNet(depth=34, num_classes=num_class, widen_factor=w_mult, dropRate=0.0).cuda()
    model.load_state_dict(gload(args.load))

    cudnn.benchmark = True
    val_loader = torch.utils.data.DataLoader(val_data_c10, batch_size=b_s, shuffle=False)

    
    validate(val_loader, model)


def validate(val_loader, model):

    model.eval()
    
    top1_a = AverageMeter()

    adver = LinfSPSAAttack(model, eps = 8./255., max_batch_size=8192, nb_iter=args.iter)
    for i, (xB, yB) in enumerate(val_loader):
            xB, yB = xB.cuda(), yB.cuda()
            #xB_adv = adversary_test.perturb(xB, yB).cuda().detach()
            xB_adv = adver.perturb(xB, yB).detach()
            with torch.no_grad():

                print(torch.max(torch.abs(xB_adv-xB)).item()*(255./8.))
                yM_a = model(xB_adv)


                prec1_a = accuracy(yM_a, yB)
               
                top1_a.update(prec1_a.item(), yB.size(0))
             
            print(top1_a.avg, (i+1)*b_s)
            wandb.log({'Avg afv acc': top1_a.avg}, step=(i+1)*b_s)
    print(top1_a.avg)

    

if __name__ == '__main__':
    main()
