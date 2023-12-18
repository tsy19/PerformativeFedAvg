#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prox_param', type=float, default=0.1,
                        help="mu")
    parser.add_argument('--epochs', type=int, default=600,
                        help="number of rounds of training")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=256,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--a', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--b', type=float, default=20,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--straggler', type=float, default=0,
                        help='straggler')
    parser.add_argument('--num_users', type=int, default=5,
                        help='straggler')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='straggler')
    parser.add_argument('--test_num_per_class', type=int, default=500,
                        help='straggler')
    parser.add_argument('--train_num_per_class', type=int, default=500,
                        help='straggler')
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")
    parser.add_argument('--seed', type=int, default=43, help='random seed')
    parser.add_argument('--static', type=str, default="dynamic", help='random seed')
    args = parser.parse_args()
    return args
