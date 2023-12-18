#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import json
import random
import pickle
import copy
import numpy as np
import torch
# from torchvision import datasets, transforms
# from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
# from sampling import cifar_iid, cifar_noniid
from torch.utils.data import TensorDataset, ConcatDataset

def split_backup(dict_users, frac = 0.8):
    dict_back = {}
    dict_beta = {}
    for i in dict_users.keys():
        dict_users[i] = list(dict_users[i])
        end = max(int(frac * len(dict_users[i])), 1)
        dict_back[i] = dict_users[i][end:]
        dict_users[i] = dict_users[i][:end]
        dict_beta[i] = 0
    return dict_users, dict_back, dict_beta

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    print('load' + args.dataset)

    train_path = os.path.join(os.path.dirname(os.getcwd()), "data/" + args.dataset + "/data/train/mytrain.pt")
    test_path = os.path.join(os.path.dirname(os.getcwd()), "data/" + args.dataset + "/data/test/mytest.pt")
    # # user_group_path = os.path.join(os.path.dirname(os.getcwd()), "data/" + args.dataset + "/data/train/user_groups.json")
    # with open(user_group_path, 'rb') as inf:
    #     user_groups = json.load(inf)
    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)

    num_classes = args.num_classes
    train_by_class = {i: [] for i in range(num_classes)}
    for item in train_dataset:
        train_by_class[item[1].item()].append(item[0])
    test_by_class = {i: [] for i in range(num_classes)}
    for item in test_dataset:
        test_by_class[item[1].item()].append(item[0])

    #downsample testset and keep it the same for all clients

    test_by_class = [downsample_images(test_by_class, num_images=args.test_num_per_class, seed=i+args.seed) for i in range(args.num_users)]
    train_by_client = []
    for client in range(args.num_users):
        train_client = downsample_images(train_by_class, num_images=args.train_num_per_class, seed=client + args.seed * args.num_users)
        total_num = num_classes * args.train_num_per_class
        x = []
        y = []
        for key, item in train_client.items():
            x.append(item.tensors[0])
            y.append(item.tensors[1])
        x = torch.cat(x)
        y = torch.hstack(y)

        train_client = TensorDataset(x, 1 / total_num * torch.ones((y.shape[0])), y)
        if args.static == "static":
            init_pi = torch.zeros((args.num_classes, ))
        else:
            parameters = [5 * (0.5**i) for i in range(10)] 
#             parameters = np.arange(10, 0, -1)
            t = parameters[0]
            parameters[0] = parameters[client]
            parameters[client] = t
            samples = [np.random.exponential(param) for param in parameters]
            
            init_pi = torch.tensor(samples)
#             init_pi = torch.tensor(parameters)
            print(init_pi)
        train_client = TensorDataset(train_client.tensors[0],
                                          adjust_weights_by_pi(train_client, init_pi, args),
                                          train_client.tensors[2])
        train_by_client.append(train_client)

    return train_by_client, test_by_class



def adjust_weights_by_pi(dataset, pi, args):
    # return pi.unsqueeze(1).repeat(1, args.train_num_per_class).flatten() / args.train_num_per_class
    temp = 1.0 - torch.clone(pi)
#     temp = torch.clone(pi) 
    min = torch.min(temp)
    temp -= min
    temp = torch.exp(0.5 * temp.unsqueeze(1).repeat(1, args.train_num_per_class).flatten())
    return temp

def downsample_images(dict_with_tensors, num_images=100, seed=0):
    random.seed(seed)
    downsampled_dict = {}
    for key, tensor_list in dict_with_tensors.items():
        num_images_to_sample = min(num_images, len(tensor_list))
        sampled_images = random.sample(tensor_list, num_images_to_sample)
        temp = torch.stack(sampled_images)
        downsampled_dict[key] = TensorDataset(temp, (key * torch.ones((temp.shape[0],))).to(dtype=torch.long))

    return downsampled_dict

def compute_group_weights(user_groups):
    weights = {}
    sum = 0
    for key, value in user_groups.items():
        sum += len(value)
        weights[int(key)] = len(value)
    for key in weights.keys():
        weights[key] /= sum
    return weights

def average_weights(ls, global_weights):
    """
    Returns the average of the weights.
    """
    sum_ws = 0
    w_avg = copy.deepcopy(ls[0][0])
    for key in w_avg.keys():
        if 'num_batches_tracked' in key:
            continue
        w_avg[key] *= ls[0][1]
    sum_ws += ls[0][1]
    for i in range(1, len(ls)):
        for key in w_avg.keys():
            if 'num_batches_tracked' in key:
                continue
            w_avg[key] += (ls[i][0][key] * ls[i][1])
        sum_ws += ls[i][1]
    for key in w_avg.keys():
        if 'num_batches_tracked' in key:
            continue
        w_avg[key] /= sum_ws

    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    print(f'    Mu     : {args.prox_param}')
    print(f'    Straggler     : {args.straggler}')
    return
