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

from helper import accuracy, pgd_attackLinf, grad_trades
from helper import AverageMeter, find_prob, find_prob2, kl, kl_full, kl2, kl2_full, cross_entropy, cross_entropy_full, criterion, criterion_full
from helper import train_data_c10, train_data_c10_aug2, val_data_c10, train_data_c100, train_data_c100_aug2, val_data_c100
from helper import train_data_c100_aug2, train_data_c100_aug3, train_data_c100_aug4, train_data_c100_aug5

from torch.optim.lr_scheduler import MultiStepLR
from WideResNet import WideResNet
from pathlib import Path
from autoattack import AutoAttack
#forward_pass= torch.load("apricot-flower-10")
parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str,
                    choices=["a", "trades", "mart", "our", "trades_kl2", "mart_kl2", "our_old", "our_2", "our_const", "mart_2", "trades_back", "basic_const"])
num_class = 10
w_mult= 1
model = WideResNet(depth=34, num_classes=num_class, widen_factor=w_mult, dropRate=0.0)
model.cuda()
model.load_state_dict(torch.load("apricot-flower-10"))
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
test_loader = torch.utils.data.DataLoader(train_data_c10, batch_size=128, shuffle=True)
global best_test
losses_n = AverageMeter()
top1_n = AverageMeter()
top1p_n = AverageMeter()
losses_a = AverageMeter()
top1_a = AverageMeter()
top1p_a = AverageMeter()

losses_mart_part2 = AverageMeter()
losses_martkl2_part2 = AverageMeter()

losses_r = AverageMeter()
losses_r_2 = AverageMeter()
losses_fin= AverageMeter()
losses_n_los = AverageMeter()

model.eval()
for i, (xB, yB) in enumerate(test_loader):
            xB, yB = xB.cuda(), yB.cuda()
            #xB_adv = adversary_test.perturb(xB, yB).cuda().detach()
            xB_adv = adversary.run_standard_evaluation( xB, yB, bs=128)
            with torch.inference_mode():

                yM_n = model(xB)

                g_n = find_prob(yM_n, yB)

                yM_a = model(xB_adv)
                g_a = find_prob(yM_a, yB)



                g_a_sec = find_prob2(yM_a, yB)



                if i % 20 == 0:
                    print(i)


                loss_a_oth = torch.mean(-1.*torch.log(1.-g_a_sec+1e-3))
                loss_n = criterion(yM_n, yB)
                loss_n_full = criterion_full(yM_n, yB)
                loss_a = criterion(yM_a, yB)
                loss_r = kl(yM_n, yM_a)
                loss_r_full = kl_full(yM_n, yM_a)


                loss_r_2 = kl2(g_n, g_a)
                loss_r_2_full = kl2_full(g_n, g_a)

                loss_mart_part2 = torch.mean(loss_r_full*(1.-g_n))
                loss_martkl2_part2 = torch.mean(loss_r_2_full*(1.-g_n))


                n_los = torch.mean(-nn.ReLU()(g_n-g_a)*loss_n_full)
                n_los_2 = torch.mean(-(g_n-g_a)*loss_n_full)

                if args.loss == "a":
                    loss_fin = loss_a
                elif args.loss == "trades":
                    loss_fin = loss_n + lam*loss_r
                elif args.loss == "trades_back":
                    loss_fin = loss_n + lam*loss_r
                elif args.loss == "tradeskl2":
                    loss_fin = loss_n + lam*loss_r_2
                elif args.loss == "mart":
                    loss_fin = loss_a+loss_a_oth + lam*loss_mart_part2
                elif args.loss == "mart_2":
                    loss_fin = loss_a+loss_a_oth + lam*loss_mart_part2
                elif args.loss == "martkl2":
                    loss_fin = loss_a+loss_a_oth + lam*loss_martkl2_part2
                elif args.loss == "our":
                    loss_fin = loss_a+lam2*loss_a_oth+lam*n_los
                elif args.loss == "our_old":
                    loss_fin = loss_n+lam*loss_r_2
                elif args.loss == "our_2":
                    loss_fin = loss_a+loss_a_oth+lam*n_los_2
                elif args.loss == "our_const":
                    loss_fin = loss_a+loss_a_oth-lam*n_los
                elif args.loss == "basic_const":
                    loss_fin = loss_a-lam*loss_n

                prec1_n = accuracy(yM_n, yB)
                prec1p_n = torch.mean(g_n)
                losses_n.update(loss_n.item(), yB.size(0))
                top1_n.update(prec1_n.item(), yB.size(0))
                top1p_n.update(prec1p_n.item(), yB.size(0))

                prec1_a = accuracy(yM_a, yB)
                prec1p_a = torch.mean(g_a)
                losses_a.update(loss_a.item(), yB.size(0))
                top1_a.update(prec1_a.item(), yB.size(0))
                top1p_a.update(prec1p_a.item(), yB.size(0))

                losses_r.update(loss_r.item(), yB.size(0))
                losses_r_2.update(loss_r_2.item(), yB.size(0))
                losses_mart_part2.update(loss_mart_part2.item(), yB.size(0))
                losses_martkl2_part2.update(loss_martkl2_part2.item(), yB.size(0))
                losses_n_los.update(n_los.item(), yB.size(0))
                losses_fin.update(loss_fin.item(), yB.size(0))
                
                print("test a-errror", 100.0-top1_a.avg,'Best Test Error', 100.0-best_test)
