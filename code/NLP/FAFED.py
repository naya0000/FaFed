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

import torch.nn as nn
from data import *
import math

def FAFED(train_data, corpus, model, args, device):

    Iteration = 0
    rank = dist.get_rank()
    size = dist.get_world_size()

    model_ = copy.deepcopy(model)
    model.train()
    model_.train()

    m_t = copy.deepcopy([i * 0 for i in list(model.state_dict().values())])
    v_t = copy.deepcopy([i * 0 for i in list(model.state_dict().values())])
    H_t = copy.deepcopy([i * 0 for i in list(model.state_dict().values())])

    total_loss = 0
    ntokens = len(corpus.dictionary)
    criterion = nn.CrossEntropyLoss()

    # Initialization
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
        data, targets = data.to(device), targets.to(device)

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                m_t[i].copy_(param.grad.data)
                dist.all_reduce(m_t[i], op=dist.ReduceOp.SUM)
                m_t[i].div_(size)

                param.data.add_(m_t[i], alpha=-args.lr)

                v_t[i].copy_(param.grad.data**2)
                dist.all_reduce(v_t[i], op=dist.ReduceOp.SUM)
                v_t[i].div_(size)

                H_t[i].copy_(torch.sqrt(v_t[i]) + args.rho)
        break  

    for epoch in range(args.epochs):
        hidden = model.init_hidden(args.batch_size)
        lr_rate = args.lr / (1 + epoch)**(1/3)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            x_para = copy.deepcopy(model.state_dict())

            data, targets = get_batch(train_data, i, args.bptt)
            data, targets = data.to(device), targets.to(device)

            hidden = repackage_hidden(hidden)
            model.zero_grad()
            model_.zero_grad()

            output_, hidden_ = model_(data, hidden)
            loss_ = criterion(output_.view(-1, ntokens), targets)
            loss_.backward()

            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(model_.parameters(), args.clip)

            # Update
            for i, (para1, para2) in enumerate(zip(model.parameters(), model_.parameters())):
                m_t[i].copy_(para1.grad.data + (1 - args.alpha) * (m_t[i] - args.ext * para2.grad.data))
                v_t[i].copy_(args.beta * v_t[i] + (1 - args.beta) * para1.grad.data**2)
                para1.data.add_(m_t[i] / H_t[i], alpha=-lr_rate)

            model_.load_state_dict(x_para)

            total_loss += loss.item()
            Iteration += 1

            ### Communication ############
            if Iteration % args.inLoop == 0:
                cur_loss = total_loss / args.inLoop
                loss_value = torch.tensor([cur_loss], dtype=torch.float64)
                dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                loss_lst = loss_value.tolist() 
                if rank == 0:
                    cur_loss = loss_lst[0] / size
                    print(epoch, Iteration // args.inLoop, cur_loss, math.exp(cur_loss))
                total_loss = 0
                
                for i, param in enumerate(model.parameters()):
                    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                    dist.all_reduce(m_t[i], op=dist.ReduceOp.SUM)
                    dist.all_reduce(v_t[i], op=dist.ReduceOp.SUM)

                    param.data.div_(size)
                    m_t[i].div_(size)
                    v_t[i].div_(size)
                    H_t[i].copy_(torch.sqrt(v_t[i]) + args.rho)

