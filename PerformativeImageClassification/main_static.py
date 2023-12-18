#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import numpy as np
from tqdm import tqdm

from options import args_parser
from update import LocalUpdate
from models import *
from utils import get_dataset, average_weights, exp_details, adjust_weights_by_pi
from torch.utils.data import TensorDataset

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # load dataset and user groups
    train_datasets, test_by_class = get_dataset(args)
    group_ws = {key: 1/args.num_users for key in range(args.num_users)}
    # BUILD MODEL
    if args.dataset == 'femnist':
        global_model = Twolayer(args=args)
        args.feature_len = 28*28

    exp_details(args)
    # Set the model to train and send it to device.
    global_model.to(device)

    # copy weights
    global_weights = global_model.state_dict()

    list_acc, list_loss = [], []
    global_model.eval()

    s1 = time.time()

    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_datasets[c], device=device, test_by_class=test_by_class[c])
        class_loss, class_accu = local_model.inference(model=global_model)
        list_acc.append(class_accu)
        list_loss.append(class_loss)
        print("0, ", class_accu, class_loss)
    #
    # tqdm.write('At round 0 accuracy: {}'.format(test_accuracy))
    # tqdm.write('At round 0 loss: {}'.format(test_loss))
    
    np.random.seed(args.seed)
    seeds = np.random.randint(1e4, size=(args.epochs, ))
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        global_model.train()
        np.random.seed(seeds[epoch])
        torch.manual_seed(seeds[epoch])
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_datasets[idx], device=device, test_by_class=test_by_class[c])
            w, class_accu, class_loss = local_model.update_weights(model=copy.deepcopy(global_model))
            local_weights.append((copy.deepcopy(w), group_ws[idx]))
            print("Epoch {} client {}: Accuracy {}, Loss {}".format(epoch+1, idx, class_accu, class_loss))
        if len(local_weights) >= 1:
            global_weights = average_weights(local_weights, global_weights)
            global_model.load_state_dict(global_weights)

        # for c in range(args.num_users):
        #     local_model = LocalUpdate(args=args, dataset=train_datasets[c], device=device, test_by_class=test_by_class[c])
        #     class_loss, class_accu = local_model.inference(model=global_model)
        #     list_acc.append(class_accu)
        #     list_loss.append(class_loss)
        #     new_pi = torch.softmax(1 / torch.tensor(class_loss), dim=0)
        #     train_datasets[c] = TensorDataset(train_datasets[c].tensors[0],
        #                                       adjust_weights_by_pi(train_datasets[c], new_pi, args),
        #                                       train_datasets[c].tensors[2])
        # print(epoch+1, ", ", )

    s2 = time.time()
    tqdm.write('Time: {}'.format(s2 - s1))
