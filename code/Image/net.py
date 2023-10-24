import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from functools import reduce
from operator import mul


class MNISTModel(nn.Module):
   
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(-1, 800)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  

        return F.log_softmax(x, dim=1)


class FashionMNISTModel(nn.Module):
   
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.fc1 = nn.Linear(250, 100)
        self.fc2 = nn.Linear(100, 10)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool2d(2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())

    def forward(self, x):
        x = self.maxpool(self.tanh(self.conv1(x)))
        x = self.maxpool(self.tanh(self.conv2(x)))
        x = x.view(-1, 250)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)  

        return F.log_softmax(x, dim=1)

class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flattening
        x = x.view(-1, 64 * 4 * 4)
        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)




