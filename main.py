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


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--w", type=int)
parser.add_argument("--loss", type=str,
                    choices=["a", "trades", "mart", "our", "trades_kl2", "mart_kl2", "our_old", "our_2", "our_const", "mart_2", "trades_back", "basic_const"])
parser.add_argument("--fd", type=int)
parser.add_argument("--sd", type=int)
parser.add_argument("--dataset", type=str, choices=["c10", "c100"], default="c10")
parser.add_argument("--bm", type=int, default=1, choices=[0, 1])
parser.add_argument("--aug", type=int, choices=[1, 2, 3, 4, 5], default=1)
parser.add_argument("--lr", type=float, default=.1)
parser.add_argument("--load", type=bool, default=False)



args = parser.parse_args()


l_rate = args.lr
momentum = .9
wd = 5e-4
b_s = 128
w_mult= args.w


PATH = 'models/i/'
PATH_load = 'models/f/35'

#Try soft labels


import wandb
wandb.init(project="SMART", name=args.name)
wandb.config.opt = "SGD"
wandb.config.lr = l_rate
wandb.config.momentum = momentum
wandb.config.wd = wd
wandb.config.b_s = b_s
wandb.config.w_mult = w_mult
wandb.config.loss_name = args.loss
wandb.config.dataset = args.dataset
wandb.config.aug = args.aug
wandb.config.bm = args.bm
wandb.config.load = args.load

num_class = -1

if args.dataset == "c10":
    num_class = 10
    if args.aug == 1:
        train_loader = torch.utils.data.DataLoader(train_data_c10, batch_size=128, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_data_c10_aug2, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data_c10, batch_size=128, shuffle=False)
elif args.dataset == "c100":
    num_class = 100
    if args.aug == 1:
        train_loader = torch.utils.data.DataLoader(train_data_c100, batch_size=128, shuffle=True)
    elif args.aug == 2:
        train_loader = torch.utils.data.DataLoader(train_data_c100_aug2, batch_size=128, shuffle=True)
    elif args.aug == 3:
        train_loader = torch.utils.data.DataLoader(train_data_c100_aug3, batch_size=128, shuffle=True)
    elif args.aug == 4:
        train_loader = torch.utils.data.DataLoader(train_data_c100_aug4, batch_size=128, shuffle=True)
    elif args.aug == 5:
        train_loader = torch.utils.data.DataLoader(train_data_c100_aug5, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data_c100, batch_size=128, shuffle=False)
else:
    print("unknown dataset")
    exit()

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

    model.train()

    for i, (xB, yB) in enumerate(train_loader):
            xB, yB = xB.cuda(), yB.cuda()
            
            if args.bm == 0:                                                                                              #UNCOMMENT
                model.eval()
            
            if args.loss == "trades" or args.loss == "trades_back":
                xB_adv = pgd_attackLinf(model, xB, yB, eps=8./255., alpha=2./255., nSteps=10, rand=True, grad_func=grad_trades)
                #print('trades', torch.mean(torch.abs(xB_adv-xB)))
                #exit()
            elif args.loss == "tradeskl2":
                print("not implemented")
                exit()
            else:
                #xB_adv = adversary.perturb(xB, yB).detach().cuda()
                xB_adv = pgd_attackLinf(model, xB, yB, eps=8./255., alpha=2./255., nSteps=10, rand=True)
                #print('a', torch.mean(torch.abs(xB_adv-xB)))
                #exit()
            model.train()
            if args.loss == "a":
                with torch.no_grad():
                    model.eval()
                    yM_n = model(xB)
                    model.train()
            else:
                yM_n = model(xB)

            g_n = find_prob(yM_n, yB)

            yM_a = model(xB_adv)



            g_a = find_prob(yM_a, yB)



            g_a_sec = find_prob2(yM_a, yB)

            g_n_d = g_n.detach()
            g_a_d = g_a.detach()


            if i % 20 == 0:
                print(i)


            loss_a_oth = torch.mean(-1.*torch.log(1.-g_a_sec+1e-3))
            loss_n = criterion(yM_n, yB)
            loss_n_full = criterion_full(yM_n, yB)
            loss_a = criterion(yM_a, yB)
            loss_r = kl(yM_n, yM_a)
            loss_r_trades = nn.KLDivLoss(size_average=False)(F.log_softmax(yM_a, dim=1),
                                           F.softmax(yM_n, dim=1))
            loss_r_full = kl_full(yM_n, yM_a)


            loss_r_2 = kl2(g_n, g_a)
            loss_r_2_trades = kl2(g_n.detach(), g_a)
            loss_r_2_full = kl2_full(g_n, g_a)

            loss_mart_part2 = torch.mean(torch.sum(nn.KLDivLoss(reduction='none')(F.log_softmax(yM_a, dim=1),
                                           F.softmax(yM_n, dim=1)), dim=1)*(1.-g_n))
            loss_mart_2_part2 = torch.mean(torch.sum(nn.KLDivLoss(reduction='none')(F.log_softmax(yM_a, dim=1),
                                           F.softmax(yM_n, dim=1)), dim=1)*(1.-g_n.detach()))
            loss_martkl2_part2 = torch.mean(loss_r_2_full*(1.-g_n))


            n_los = torch.mean(-nn.ReLU()(g_n_d-g_a_d)*loss_n_full)
            n_los_2 = torch.mean(-(g_n_d-g_a_d)*loss_n_full)

            if args.loss == "a":
                loss_fin = loss_a
                if i == 0:
                    print("loss: a")
            elif args.loss == "trades":
                loss_fin = loss_n + lam*loss_r_trades/yB.size(0)
            elif args.loss == "trades_back":
                loss_fin = loss_n + lam*loss_r
            elif args.loss == "tradeskl2":
                loss_fin = loss_n + lam*loss_r_2_trades
            elif args.loss == "mart":
                loss_fin = loss_a+loss_a_oth + lam*loss_mart_part2
            elif args.loss == "mart_2":
                loss_fin = loss_a+loss_a_oth + lam*loss_mart_2_part2
            elif args.loss == "martkl2":
                loss_fin = loss_a+loss_a_oth + lam*loss_martkl2_part2
            elif args.loss == "our":
                loss_fin = loss_a+lam2*loss_a_oth+lam*n_los
            elif args.loss == "our_2":
                loss_fin = loss_a+loss_a_oth+lam*n_los_2
            elif args.loss == "our_old":
                loss_fin = loss_n+lam*loss_r_2
            elif args.loss == "our_const":
                loss_fin = loss_a+loss_a_oth-lam*loss_n
            elif args.loss == "basic_const":
                loss_fin = loss_a-lam*loss_n

            with torch.no_grad():
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


        # compute gradient and do SGD step

            optimizer.zero_grad()
            loss_fin.backward()
            optimizer.step()

    wandb.log({'epoch': epoch, 'Train n loss': losses_n.avg, 'Train n error': 100.0-top1_n.avg, 'Train n p error': 100.0-top1p_n.avg,
        'Train a loss': losses_a.avg, 'Train a error': 100.0-top1_a.avg, 'Train a p error': 100.0-top1p_a.avg, 'Train Loss r': losses_r.avg,
              'Train loss fin':  losses_fin.avg, 'Train loss mart_part2': losses_mart_part2.avg, 'Train loss r_2': losses_r_2.avg, 'Train loss martkl2_part2': losses_martkl2_part2.avg,
              'Train loss n_los':  losses_n_los.avg}, step=epoch)
    print('Epoch: [{0}]\t'.format(epoch))

def validate(val_loader, model, epoch):
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

    for i, (xB, yB) in enumerate(val_loader):
            xB, yB = xB.cuda(), yB.cuda()
            #xB_adv = adversary_test.perturb(xB, yB).cuda().detach()
            xB_adv = pgd_attackLinf(model, xB, yB, eps=8./255., alpha=.8/255., nSteps=20, rand=False)
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

    if best_test < top1_a.avg:
        best_test = top1_a.avg
        #torch.save(model.state_dict(), PATH+str(epoch))
        print('saved')
    wandb.log({'epoch': epoch, 'Test n loss': losses_n.avg, 'Test n error': 100.0-top1_n.avg, 'Test n p error': 100.0-top1p_n.avg,
        'Test a loss': losses_a.avg, 'Test a error': 100.0-top1_a.avg, 'Test a p error': 100.0-top1p_a.avg, 'Test Loss r': losses_r.avg,
              'Test loss fin':  losses_fin.avg, 'Test loss mart_part2': losses_mart_part2.avg, 'Test loss r_2': losses_r_2.avg, 'Test loss martkl2_part2': losses_martkl2_part2.avg,
              'Test loss n_los':  losses_n_los.avg, 'Best Test Error': 100.0-best_test}, step=epoch)
    print('Epoch: [{0}]\t'.format(epoch))

if __name__ == '__main__':
    main()
