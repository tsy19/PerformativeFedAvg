#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import copy
import time
from torch import nn
from torch.utils.data import DataLoader
from FedProxOptimizer import FedProxOptimizer
from torch.optim import Adam
from utils import adjust_weights_by_pi
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import LinearLR, LambdaLR
class LocalUpdate(object):
    def __init__(self, args, dataset, device, test_by_class=None):
        self.args = args
        self.dataset = dataset
        self.trainloader = DataLoader(dataset,
                                      batch_size=self.args.local_bs, shuffle=True)
        self.test_by_class = test_by_class
        self.device = device
        # self.criterion = nn.NLLLoss().to(self.device)
        # if args.model == 'logistic' or args.model == 'twolayer':
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def update_weights(self, model):
        # Set mode to train model
        model.train()
        #mu=0 is exactly FedAvg, not FedProx
        optimizer = FedProxOptimizer(model.parameters(),
                                     # lr=self.args.lr,
                                     lr=1,
                                     mu=0,
                                     momentum=self.args.momentum,
                                     nesterov=False,
                                     weight_decay=0)
        lr_func = lambda epoch: self.args.a / (self.args.b + self.args.current_epoch * self.args.local_ep + epoch)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_func)
        early_stop = self.args.local_ep

        #for struggler
        if torch.rand(1, device=self.device) < self.args.straggler:
            early_stop = int(torch.torch.rand(1) * self.args.local_ep)
            while early_stop == 0:
                early_stop = int(torch.torch.rand(1) * self.args.local_ep)
        pi_list = torch.zeros((self.args.local_ep, self.args.num_classes))
        for iter in range(self.args.local_ep):
            model.train()
            if iter >= early_stop:
                break
            for batch_idx, (images, sample_weights, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device, dtype=torch.float), labels.to(device=self.device, dtype=torch.long)
                sample_weights = sample_weights.to(self.device)
                if self.args.dataset == "femnist":
                    images = images.flatten(start_dim=1)
                model.zero_grad()
                log_probs = model(images)
                loss = (self.criterion(log_probs, labels) * sample_weights).mean()
                # loss_by_sample = nn.functional.cross_entropy(log_probs, labels, reduction='none')
                # loss = loss_by_sample * sample_weights * self.args.train_num_per_class * self.args.num_users
                loss.backward()
                optimizer.step()
            scheduler.step()
#             print("iter {}, lr {}".format(iter, scheduler.get_last_lr()[0]))
            class_loss, class_accu = self.inference(model)
            # new_pi = torch.softmax(1 / torch.tensor(class_loss), dim=0)
            if self.args.static != "static":
#                 pi_list[iter, :] = class_loss
                pi_list[iter, :] = class_accu
                self.dataset = TensorDataset(self.dataset.tensors[0],
                                      adjust_weights_by_pi(self.dataset, class_accu, self.args),
                                      self.dataset.tensors[2])
                self.trainloader = DataLoader(self.dataset,
                                          batch_size=self.args.local_bs, shuffle=True)
        w = copy.deepcopy(model.state_dict())
#         print(pi_list)
        return w, class_accu, class_loss, pi_list, scheduler.get_last_lr()[0]

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        class_loss = []
        class_accu = []
        for key, item in self.test_by_class.items():
            images = item.tensors[0].to(self.device, dtype=torch.float)
            labels = item.tensors[1].to(self.device, dtype=torch.long)
            outputs = model(images)
            # labels = labels.reshape(outputs.shape)
            class_loss.append(self.criterion(outputs, labels).item())
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.reshape(labels.shape)
            class_accu.append(torch.sum(pred_labels == labels).item() / images.shape[0])

        return torch.tensor(class_loss), torch.tensor(class_accu)
#