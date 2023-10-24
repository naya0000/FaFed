import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.optim as optim
import random
import torch.nn.functional as F
import numpy as np
import argparse
import copy

def FAFED(train_set, test_set, model, args, device):

    model_ = copy.deepcopy(model)
    Iteration = 0

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer1 = optim.SGD(model_.parameters(), lr=args.lr)

    rank = dist.get_rank()
    size = dist.get_world_size()

    m_t = copy.deepcopy([i * 0 for i in list(model.state_dict().values())])
    v_t = copy.deepcopy([i * 0 for i in list(model.state_dict().values())])
    H_t = copy.deepcopy([i * 0 for i in list(model.state_dict().values())])

    for siter, (data, target) in enumerate(train_set):
        # Initialization
        optimizer.zero_grad()
        data   = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        for i, param in enumerate(model.parameters()):
            m_t[i] = torch.clone(param.grad.data).detach()   
            dist.all_reduce(m_t[i], op=dist.ReduceOp.SUM)
            m_t[i].div_(size)
            param.data.add_(m_t[i], alpha=-args.lr)
            v_t[i] = torch.clone(param.grad.data**2).detach()
            dist.all_reduce(v_t[i], op=dist.ReduceOp.SUM)
            v_t[i].div_(size)
            H_t[i] = torch.sqrt(v_t[i]) + args.rho
        break  

    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for siter, (data, target) in enumerate(train_set):
            model.train()
            x_para = copy.deepcopy(model.state_dict())

            data   = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            output_ = model_(data)
            loss_ = F.nll_loss(output_, target)
            loss_.backward()

            # Update
            for i, (para1, para2) in enumerate(zip(model.parameters(), model_.parameters())):
                m_t[i] = para1.grad.data + (1 - args.alpha) * (m_t[i] - args.ext * para2.grad.data)
                v_t[i] = args.beta * v_t[i] + (1 - args.beta) * para1.grad.data**2
                para1.data.add_(m_t[i] / H_t[i], alpha=-args.lr)

            optimizer.zero_grad()
            optimizer1.zero_grad()
            model_.load_state_dict(x_para)

            ###### Printing loss ####
            if Iteration % args.inLoop == 0:
                #### testing  #######
                model.eval()
                correct_cnt, ave_loss = 0, 0
                total_cnt = 0
                for batch_idx, (data, target) in enumerate(test_set):

                    data   = data.to(device)
                    target = target.to(device)

                    out = model(data)
                    _, pred_label = torch.max(out.data, 1)
                    total_cnt += data.data.size()[0]
                    correct_cnt += (pred_label == target.data).sum()

                #########
                loss_value = torch.tensor([loss.item(), correct_cnt, total_cnt], dtype=torch.float64)
                dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                loss_lst = loss_value.tolist()

                if rank == 0:
                    # print(Iteration // args.inLoop, loss_value.item(), correct_cnt * 1.0 / total_cnt) 
                    print(Iteration // args.inLoop, loss_lst[0] / size, loss_lst[1] / loss_lst[2] ) 
                ###########
            Iteration += 1

            # Communication
            if Iteration % args.inLoop == 0:
                for i, param in enumerate(model.parameters()):
                    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                    dist.all_reduce(m_t[i], op=dist.ReduceOp.SUM)
                    dist.all_reduce(v_t[i], op=dist.ReduceOp.SUM)

                    param.data.div_(size)
                    m_t[i].div_(size)
                    v_t[i].div_(size)
                    H_t[i] = torch.sqrt(v_t[i]) + args.rho
