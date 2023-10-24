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
import time
import copy

def FedAvg(train_set, test_set, model, args, device):
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
	Iteration = 0
	rank = dist.get_rank()
	size = dist.get_world_size()

	for epoch in range(args.epochs):

		for siter, (data, target) in enumerate(train_set):
			model.train()

			data   = data.to(device)
			target = target.to(device)

			output = model(data)

			loss = F.nll_loss(output, target)
			loss.backward()

            # Update
			optimizer.step()
			optimizer.zero_grad()

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

			Iteration += 1

			# ### Communication ############
			if Iteration % args.inLoop == 0:
				for para in model.parameters():
					dist.all_reduce(para.data, op=dist.ReduceOp.SUM)
					para.data.div_(size)
