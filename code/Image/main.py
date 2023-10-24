import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.optim as optim
from net import *
from dist_data import *
from FedAvg import *
from FAFED import *

import random
import numpy as np
import argparse
import time

def get_default_device(idx):
    if torch.cuda.is_available():
        print("GPU available")
        return torch.device('cuda:' + str(idx))
    else:
        print("GPU NOT available")

        return torch.device('cpu')

""" Partitioning Dataset """
def partition_dataset(args):

    rank = dist.get_rank()
    
    print("args.dataset",args.dataset);

    if args.dataset == 'mnist1':
        dataset = DISTMNIST(root='./mnist1/', rank=rank, train=True, download=True, transform=transform_m)
        testset = DISTMNIST(root='./mnist1/', rank=rank, train=False, download=True, transform=transform_m)
    elif args.dataset == 'mnist2':
        dataset = HiDISTMNIST(root='./mnist2/', rank=rank, train=True, download=True, transform=transform_m)
        testset = DISTMNIST(root='./mnist1/', rank=rank, train=False, download=True, transform=transform_m)
    elif args.dataset == 'mnist':
        args.dataset = args.dataset + '1'
        dataset = DISTMNIST(root='./mnist1/', rank=rank, train=True, download=True, transform=transform_m)
        testset = DISTMNIST(root='./mnist1/', rank=rank, train=False, download=True, transform=transform_m)
    elif args.dataset == 'fmnist1':
        dataset = DISTFashionMNIST('./fashionmnist1/', rank=rank, train=True, download=True,  transform=transform_f)
        testset = DISTFashionMNIST('./fashionmnist1/', rank=rank, train=False, download=True, transform=transform_f)
    elif args.dataset == 'fmnist2':
        dataset = HiDISTFashionMNIST('./fashionmnist2/', rank=rank, train=True, download=True,  transform=transform_f)
        testset = DISTFashionMNIST('./fashionmnist2/', rank=rank, train=False, download=True, transform=transform_f)

    elif args.dataset == 'cifar1':
        dataset = DISTCIFAR10('./cifar10d/', rank=rank, train=True, download=True, transform=transform_c)
        testset = DISTCIFAR10('./cifar10d/', rank=rank, train=False, download=True, transform=transform_c)
    elif args.dataset == 'cifar2':
        dataset = HiDISTCIFAR10('./cifar10h/', rank=rank, train=True, download=True, transform=transform_c)
        testset = HiDISTCIFAR10('./cifar10dh/', rank=rank, train=False, download=True, transform=transform_c)
    else:
        raise  NotImplementedError('Unsupported Datasets!')

    train_set = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,   #For more stable results, shuffle can turn to False.
                                         num_workers=4)

    test_set = torch.utils.data.DataLoader(testset,
                                         batch_size=args.test_batch_size,
                                         # shuffle=True,   #For more stable results, shuffle can turn to False.
                                         num_workers=4)

    return train_set, test_set

""" Distributed Synchronous """
def run(rank, size, args):

    rank = dist.get_rank()
    div_pat = dist.get_world_size()

    device = get_default_device(int(rank // div_pat))

    print("args.dataset",args.dataset);

    train_set, test_set  =  partition_dataset(args)

    if args.dataset == 'mnist0' or args.dataset == 'mnist' or args.dataset == 'mnist1' or args.dataset == 'mnist2':  
        model = MNISTModel() 
    elif args.dataset == 'fmnist0' or args.dataset == 'fmnist1' or args.dataset == 'fmnist2':  
        model = FashionMNISTModel() 
    elif args.dataset == 'cifar0' or args.dataset == 'cifar1' or args.dataset == 'cifar2':  
        model = CIFARModel()     
    else:
        print("ERRORRRR")
        return

    if args.init:
        print("Wegith Init")
        fname = args.dataset + '/model.pth'
        torch.save(model.state_dict(), fname)

    else:
        print("Wegith Loaded")
        fname = args.dataset + '/model.pth'
        model.load_state_dict(torch.load(fname))


    model = model.to(device)

    if args.method == 'fedavg':
        FedAvg(train_set, test_set, model, args, device)
    elif args.method == 'fafed':
        FAFED(train_set, test_set, model, args, device) 

    else:
        print("ERRORRRR")

    print("FL done")



def init_process(rank, size, args, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(args.port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 5000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--worker-size', type=int, default=2, metavar='N',
                        help='szie of worker (default: 4)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--glr', type=float, default=0.032, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='alpha',
                        help='momentum rate alpha')
    parser.add_argument('--beta', type=float, default=0.1, metavar='alpha',
                        help='momentum rate beta')
    parser.add_argument('--ext', type=float, default=0.95, metavar='alpha',
                        help='momentum rate beta')
    parser.add_argument('--rho', type=float, default=0.1, metavar='alpha',
                        help='momentum rate rho')

    parser.add_argument('--inLoop', type=int, default=10, metavar='S',
                        help='inter loop number')
    parser.add_argument('--init', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=str, default='mnist1',
                        help='Dataset for trainig')
    parser.add_argument('--method', type=str, default='fedavg',
                        help='Dataset for trainig')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1234)')
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