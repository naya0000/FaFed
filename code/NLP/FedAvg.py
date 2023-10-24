import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.optim as optim
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import time
import copy

import torch.nn as nn
from data import *
import math

def FedAvg(train_data, corpus, model, args, device):

	Iteration = 0
	rank = dist.get_rank()
	size = dist.get_world_size()
	model.train()

	total_loss = 0
	ntokens = len(corpus.dictionary)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(args.epochs):
		hidden = model.init_hidden(args.batch_size)
		for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
			data, targets = get_batch(train_data, i, args.bptt)
			data, targets = data.to(device), targets.to(device)

			# Starting each batch, we detach the hidden state from how it was previously produced.
			# If we didn't, the model would try backpropagating all the way to start of the dataset.
			hidden = repackage_hidden(hidden)
			model.zero_grad()
			output, hidden = model(data, hidden)
			loss = criterion(output.view(-1, ntokens), targets)
			loss.backward()

			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
			for p in model.parameters():
				p.data.add_(p.grad.data, alpha=-args.lr)

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

				for para in model.parameters():
					dist.all_reduce(para.data, op=dist.ReduceOp.SUM)
					para.data.div_(size)

