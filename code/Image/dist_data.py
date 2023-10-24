import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from collections import Counter
import torch.distributed as dist

transform_c = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])


transform_m =transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])

transform_f = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(
        (0.2860,), (0.3529,)) ])


class DISTMNIST(torchvision.datasets.MNIST):

    def __init__(self, root, rank, train=True,
                 transform=None, target_transform=None, download=False):
        super(DISTMNIST, self).__init__(root, train, transform, target_transform, download)
        
        rand_number = 0
        np.random.seed(rand_number)
        # np.random.shuffle(idx)

        size = dist.get_world_size()
 
        data_sizes = [len(self.data) // size for i in range(size)]
        for i in range(len(self.data) % size):
            data_sizes[i] += 1

        start_idx = np.sum(data_sizes[:rank], dtype=int)
        end_idx = start_idx + data_sizes[rank]
        # print(data_sizes, rank, start_idx, end_idx)


        self.data = self.data[start_idx:end_idx]
        self.targets = self.targets[start_idx:end_idx]

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target

class DISTFashionMNIST(torchvision.datasets.FashionMNIST):

    def __init__(self, root, rank, train=True,
                 transform=None, target_transform=None, download=False):
        super(DISTFashionMNIST, self).__init__(root, train, transform, target_transform, download)
        
        rand_number = 0
        np.random.seed(rand_number)
        # np.random.shuffle(idx)

        size = dist.get_world_size()
 
        data_sizes = [len(self.data) // size for i in range(size)]
        for i in range(len(self.data) % size):
            data_sizes[i] += 1

        start_idx = np.sum(data_sizes[:rank], dtype=int)
        end_idx = start_idx + data_sizes[rank]
        # print(data_sizes, rank, start_idx, end_idx)


        self.data = self.data[start_idx:end_idx]
        self.targets = self.targets[start_idx:end_idx]

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target



class DISTCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, rank, train=True,
                 transform=None, target_transform=None, download=False):
        super(DISTCIFAR10, self).__init__(root, train, transform, target_transform, download)

        if train != True:        
        #     rand_number = 0
        #     np.random.seed(rand_number)
            # np.random.shuffle(idx)

            size = dist.get_world_size()
     
            data_sizes = [len(self.data) // size for i in range(size)]
            for i in range(len(self.data) % size):
                data_sizes[i] += 1

            start_idx = np.sum(data_sizes[:rank], dtype=int)
            end_idx = start_idx + data_sizes[rank]
            # print(data_sizes, rank, start_idx, end_idx)

            self.data = self.data[start_idx:end_idx]
            self.targets = self.targets[start_idx:end_idx]
        else:
            # print(self.targets[0])

            Y = torch.FloatTensor(self.targets) 
            # np.array(self.targets)
            size = dist.get_world_size()

            idx0 = (Y[:] == 0).nonzero().squeeze(1)
            idx1 = (Y[:] == 1).nonzero().squeeze(1)
            idx2 = (Y[:] == 2).nonzero().squeeze(1)
            idx3 = (Y[:] == 3).nonzero().squeeze(1)
            idx4 = (Y[:] == 4).nonzero().squeeze(1)
            idx5 = (Y[:] == 5).nonzero().squeeze(1)
            idx6 = (Y[:] == 6).nonzero().squeeze(1)
            idx7 = (Y[:] == 7).nonzero().squeeze(1)
            idx8 = (Y[:] == 8).nonzero().squeeze(1)
            idx9 = (Y[:] == 9).nonzero().squeeze(1)

            idx = torch.cat((idx0, idx1, idx2, idx3, idx4, 
                idx5, idx6, idx7, idx8, idx9), 0)

            X_train = self.data[idx]
            y_train = Y[idx]

            self.data = X_train[rank::size]
            y_train = y_train[rank::size]

            # permutation
            permutation = torch.randperm(len(self.data))
            self.data = self.data[permutation]
            y_train = y_train[permutation].to(dtype=torch.int32)

            self.targets = y_train.tolist()


            # print(self.targets[0])
            # print("--", rank, np.shape(y_train), type(self.targets))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target


class HiDISTMNIST(torchvision.datasets.MNIST):
    # (0, 1, 2, 5, 9, 29993)
    def __init__(self, root, rank, train=True,
                 transform=None, target_transform=None, download=False):
        super(HiDISTMNIST, self).__init__(root, train, transform, target_transform, download)

        half_size = int(dist.get_world_size() / 2)
        div_pat = int(rank // half_size)
        div_rank = rank - div_pat * half_size

        lst_total = [[0, 1, 2, 5, 9], [3, 4, 6, 7, 8]]
        lst = lst_total[div_pat]

        Y = self.targets.numpy() 
        idx = np.any([Y == lst[0], Y == lst[1], Y == lst[2], Y == lst[3], Y == lst[4]], axis=0)
        X_train =  self.data[idx]
        y_train =  self.targets[idx]

        # permutation
        permutation = torch.randperm(len(X_train))
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        data_sizes = [len(X_train) // half_size for i in range(half_size)]
        for i in range(len(X_train) % half_size):
            data_sizes[i] += 1

        start_idx = np.sum(data_sizes[:div_rank], dtype=int)
        end_idx = start_idx + data_sizes[div_rank]
        # print(len(X_train), div_rank, start_idx, end_idx)

        self.data = X_train[start_idx:end_idx]
        self.targets = y_train[start_idx:end_idx]

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target

class HiDISTFashionMNIST(torchvision.datasets.FashionMNIST):
    # (0, 1, 2, 5, 9, 29993)
    def __init__(self, root, rank, train=True,
                 transform=None, target_transform=None, download=False):
        super(HiDISTFashionMNIST, self).__init__(root, train, transform, target_transform, download)

        half_size = int(dist.get_world_size() / 2)
        div_pat = int(rank // half_size)
        div_rank = rank - div_pat * half_size

        lst_total = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        lst = lst_total[div_pat]

        Y = self.targets.numpy() 

        idx0 = (self.targets[:] == lst[0]).nonzero().squeeze(1)
        idx1 = (self.targets[:] == lst[1]).nonzero().squeeze(1)
        idx2 = (self.targets[:] == lst[2]).nonzero().squeeze(1)
        idx3 = (self.targets[:] == lst[3]).nonzero().squeeze(1)
        idx4 = (self.targets[:] == lst[4]).nonzero().squeeze(1)
        idx = torch.cat((idx0, idx1, idx2, idx3, idx4), 0)

        X_train = self.data[idx]
        y_train = self.targets[idx]

        self.data = X_train[div_rank::half_size]
        self.targets = y_train[div_rank::half_size]
        # print("rank", rank, div_rank)

        # permutation
        permutation = torch.randperm(len(self.data))
        self.data = self.data[permutation]
        self.targets = self.targets[permutation]

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target
############

class HiDISTCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, rank, train=True,
                 transform=None, target_transform=None, download=False):
        super(HiDISTCIFAR10, self).__init__(root, train, transform, target_transform, download)

        if train != True:        
            size = dist.get_world_size()
     
            data_sizes = [len(self.data) // size for i in range(size)]
            for i in range(len(self.data) % size):
                data_sizes[i] += 1

            start_idx = np.sum(data_sizes[:rank], dtype=int)
            end_idx = start_idx + data_sizes[rank]
            # print(data_sizes, rank, start_idx, end_idx)

            self.data = self.data[start_idx:end_idx]
            self.targets = self.targets[start_idx:end_idx]
        else:
            half_size = int(dist.get_world_size() / 2)
            div_pat = int(rank // half_size)
            div_rank = rank - div_pat * half_size

            # lst_total = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
            lst_total = [[9, 1, 2, 3, 4], [5, 6, 7, 8, 0]]

            lst = lst_total[div_pat]

            Y = torch.FloatTensor(self.targets) 

            idx0 = (Y[:] == lst[0]).nonzero().squeeze(1)
            idx1 = (Y[:] == lst[1]).nonzero().squeeze(1)
            idx2 = (Y[:] == lst[2]).nonzero().squeeze(1)
            idx3 = (Y[:] == lst[3]).nonzero().squeeze(1)
            idx4 = (Y[:] == lst[4]).nonzero().squeeze(1)
            idx = torch.cat((idx0, idx1, idx2, idx3, idx4), 0)

            X_train = self.data[idx]
            y_train = Y[idx]

            self.data = X_train[div_rank::half_size]
            y_train = y_train[div_rank::half_size]

            # permutation
            permutation = torch.randperm(len(self.data))
            self.data = self.data[permutation]
            y_train = y_train[permutation].to(dtype=torch.int32)

            self.targets = y_train.tolist()

            # print(self.targets[0])
            # print("--", rank, np.shape(y_train), type(self.targets))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target
