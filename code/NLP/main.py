
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms

from FedAvg import *
from FAFED import *

import random
import numpy as np
import argparse
import time
import math
import torch.nn as nn
import torch.onnx

from data import *
from model import RNNModel

def get_default_device(idx):
    # print(idx)
    if torch.cuda.is_available():
        print("GPU available")
        return torch.device('cuda:' + str(idx))
    else:
        print("GPU NOT available")

        return torch.device('cpu')

""" Distributed Synchronous """
def run(rank, size, args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()


    rank = dist.get_rank()
    div_pat = dist.get_world_size() / 4
    device = get_default_device(int(rank // div_pat))

    # Load data
    corpus = Corpus(args.data, size, rank)
    print("rank:", rank, "Train: ", len(corpus.train))

    train_data = batchify(corpus.train, args.batch_size)
    ntokens = len(corpus.dictionary)
    model = RNNModel('LSTM', ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).to(device)


    if args.method == 'fedavg':
        FedAvg(train_data, corpus, model, args, device)
    elif args.method == 'fafed':
        FAFED(train_data, corpus, model, args, device) 
    else:
        print("ERRORRRR")

    print("FL done")


def init_process(rank, size, args, fn, backend='gloo'):
# def init_process(rank, size, args, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(args.port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--worker-size', type=int, default=20, metavar='N',
                        help='szie of worker (default: 4)')
    parser.add_argument('--emsize', type=int, default=650,
                    help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=650,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--glr', type=float, default=0.032, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='alpha',
                        help='momentum rate alpha')
    parser.add_argument('--beta', type=float, default=0.1, metavar='alpha',
                        help='momentum rate beta')
    parser.add_argument('--ext', type=float, default=1, metavar='alpha',
                        help='momentum rate beta')
    parser.add_argument('--rho', type=float, default=0.1, metavar='alpha',
                        help='momentum rate rho')
    parser.add_argument('--inLoop', type=int, default=10, metavar='S',
                        help='inter loop number')
    parser.add_argument('--init', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
    parser.add_argument('--method', type=str, default='fedavg',
                        help='Dataset for trainig')
    parser.add_argument('--seed', type=int, default=1111, metavar='S',
                        help='random seed (default: 1111)')
    parser.add_argument('--port', type=int, default=29505, metavar='S',
                        help='random seed (default: 29505)')
    args = parser.parse_args()
    print(args)

    size = args.worker_size 
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, args, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()