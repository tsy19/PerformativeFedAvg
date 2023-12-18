#!/usr/bin/env python
# coding: utf-8
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rc
# import matplotlib
# import matplotlib.ticker as mtick
# from matplotlib.ticker import FormatStrFormatter
# from matplotlib.lines import Line2D
import math
import time
import random
from types import prepare_class


def run(num_time_steps, E, fraction, with_, eps_var, mu_var, C=10, h="", algorithm="fedavg"):
    start_time = time.time()
    mu_p = 1.0
    gamma = 1.0
    beta  = 1.0
    L     = 1.0

    # eps_var = 0.5
    eps_avg = 0.9
    mu_avg = 10
    # mu_var = 5
    num_clients = 25
    print(with_)
    np.random.seed(seed=0)
    if with_=="with_" or with_=="full_":
        p_ls = np.random.rand(num_clients)
        p_ls = p_ls / np.sum(p_ls)
    else: 
        p_ls = np.ones((num_clients,))
        p_ls = p_ls / np.sum(p_ls)
    eps = np.random.rand(num_clients)
    eps = np.sqrt(eps_var / np.var(eps)) * eps
    eps_ls = eps + eps_avg - eps @ p_ls
    mu = np.random.rand(num_clients)
    mu = np.sqrt(mu_var / np.var(mu)) * mu
    mu = mu + mu_avg - mu @ p_ls
    print(mu, mu @ p_ls)
    print(eps_ls, eps_ls @ p_ls)
    print(p_ls)
    eps_mean = eps_ls @ p_ls
    mu_mean = mu @ p_ls
    sigma = 0.1
    K = int(num_clients * fraction)

    # performatively stable point
    mu_PS = mu_mean / (1.0-eps_mean)
    print(mu_PS)
    num_rep = 5
    theta_init = mu_mean
    # repeat

    def get_K_samples(K, p_ls, with_):
        if with_=="with_":
            num_samples = np.random.multinomial(K, p_ls).tolist()
            sample_ls = []
            for i in range(num_clients):
                if num_samples[i] > 0:
                    sample_ls.extend([i]*num_samples[i])

            return sample_ls
        elif with_=="full_":
            return list(range(p_ls.shape[0]))
        else:
            sample_ls = random.sample(range(num_clients), K)
            return sample_ls

    # recording deployed theta and theta iterates
    record_theta = np.zeros([num_rep, num_time_steps])
    record_theta_avg = np.zeros([num_rep, num_time_steps])


    silence = False

    # repeat
    for rep in range(num_rep):
        if not silence:
            print(rep)

        # initialize training
        theta = theta_init
        local_theta = [theta]*num_clients

        # total time count
        for t in range(num_time_steps):
    #         print(t)
            if t == 0:

                # initial classifiers
                record_theta[rep,t] = theta
                record_theta_avg[rep,t] = theta

                if not silence:
                    print(round(abs(theta - mu_PS), 4))
                continue

            # local stepsize schedule
            stepsize = 100/(t+10**4)
            #stepsize = 0.2

            if (t+1)%(10**4) == 0 and not silence:
                print(t, round(abs(theta - mu_PS), 4))

            # Aggregate and Broadcast
            if (t+1)%E == 0:
                sample_ls = get_K_samples(K, p_ls, with_)
#                 print(sample_ls)
                for _ in range(C):
                    theta = sum([local_theta[i] for i in sample_ls])
                theta = theta/K
                for _ in range(C):
                    local_theta = [theta]*num_clients

            # Sampling Clients, Local Updates
            for i in range(num_clients):
                # mean of this mixture component
                this_mu = mu[i] + eps_ls[i]*local_theta[i]

                # random sample
                s = np.random.normal(this_mu, sigma/100, 1)[0]
    #             if (t+1)%100==0:
    #                 print(t+1, i, mu_PS, this_mu, s - local_theta[i], local_theta[i])
                if algorithm == "fedprox":
                    local_theta[i] = local_theta[i] + stepsize * (s - local_theta[i]) - mu_p * stepsize * (s - theta)
                else:
                    local_theta[i] = local_theta[i] + stepsize * (s - local_theta[i])

            # record state of current step
            record_theta[rep,t] = theta
            record_theta_avg[rep,t] = sum([p_ls[i]*local_theta[i] for i in range(num_clients)])

    print("total time", time.time() - start_time)

    np.save(with_+"E"+str(E)+"N"+str(num_clients)+"K"+str(K)+h+".npy", record_theta_avg)


# run(100000, 5, 1, "with_")
# run(100000, 1, 1, "with_")
# run(100000, 5, 1, "with_")
# run(100000, 20, 1, "with_")

#
# eps_var = 0.01
# mu_var = 0.0
#server_step = 6000
# run(server_step * 5, 5, 1, "without_", eps_var, mu_var)
# run(server_step * 5, 5, 0.2, "without_", eps_var, mu_var)
# run(server_step * 5, 5, 0.8, "without_", eps_var, mu_var)
# run(server_step * 5, 5, 0.6, "without_", eps_var, mu_var)
# run(server_step * 5, 5, 0.2, "with_", eps_var, mu_var)
# run(server_step * 5, 5, 0.8, "with_", eps_var, mu_var)
# run(server_step * 5, 5, 0.6, "with_", eps_var, mu_var)
# run(server_step * 5, 5, 0.8, "without_", eps_var, mu_var)
# run(server_step * 1, 1, 0.8, "without_", eps_var, mu_var)
# run(server_step * 10, 10, 0.8, "without_", eps_var, mu_var)
# run(server_step * 20, 20, 0.8, "without_", eps_var, mu_var)
# run(server_step * 50, 50, 0.8, "without_", eps_var, mu_var)
# run(server_step * 5, 5, 0.2, "without_", eps_var, mu_var)
# run(server_step * 5, 5, 0.4, "without_", eps_var, mu_var)
# run(server_step * 5, 5, 0.6, "without_", eps_var, mu_var)
# run(server_step * 5, 5, 0.8, "without_", eps_var, mu_var)
# run(server_step * 5, 5, 1, "full_", eps_var, mu_var)
# run(server_step * 5, 1, 1, "full_", eps_var, mu_var)
# run(server_step * 5, 1, 1, "full_", eps_var, mu_var)

# run(server_step * 5, 5, 1, "full_", eps_var, mu_var)

# run(server_step * 5, 5, 1, "with_", eps_var, mu_var)
# run(server_step * 5, 1, 1, "full_", eps_var, mu_var)
# run(server_step * 5, 10, 1, "with_", eps_var, mu_var)
# run(server_step * 5, 20, 1, "with_", eps_var, mu_var)

#eps_var = 0.0
#mu_var = 0.0
# run(server_step * 10, 10, 1, "full_", eps_var, mu_var, "h")
# run(server_step * 5, 5, 1, "full_", eps_var, mu_var, "h")
# run(server_step * 1, 1, 1, "full_", eps_var, mu_var, "h")
# run(server_step * 1, 1, 0.8, "with_", eps_var, mu_var, "h")
#run(server_step * 10, 10, 0.8, "with_", eps_var, mu_var, "nh")
# run(server_step * 20, 20, 0.8, "with_", eps_var, mu_var, "h")
# run(server_step * 1, 1, 0.8, "without_", eps_var, mu_var, "h")
#run(server_step * 10, 10, 0.8, "without_", eps_var, mu_var, "nh")
# run(server_step * 20, 20, 0.8, "without_", eps_var, mu_var, "h")
# run(server_step * 50, 50, 0.8, "with_", eps_var, mu_var, "h")
# run(server_step * 50, 50, 0.8, "without_", eps_var, mu_var, "h")
# run(server_step * 5, 5, 0.8, "without_", eps_var, mu_var, "h")

# run(server_step * 5, 1, 1, "without_", eps_var, mu_var, "h")
# run(server_step * 5, 10, 1, "without_", eps_var, mu_var, "h")
# run(server_step * 5, 20, 1, "without_", eps_var, mu_var, "h")

eps_var = 0.6
mu_var = 6
server_step = 300000
# run(server_step * 5, 5, 0.8, "without_", eps_var, mu_var, "h_mu")
# run(server_step * 10, 10, 0.8, "with_", eps_var, mu_var, "h_mu")
# run(server_step * 10, 10, 1, "full_", eps_var, mu_var, "h")
# run(server_step * 5, 5, 1, "full_", eps_var, mu_var, "h")
# run(server_step * 1, 1, 1, "full_", eps_var, mu_var, "h")
# run(server_step * 1, 1, 0.8, "with_", eps_var, mu_var, "h")
# run(server_step * 20, 20, 0.8, "with_", eps_var, mu_var, "h")
# run(server_step * 1, 1, 0.8, "without_", eps_var, mu_var, "h")
#run(server_step * 5, 5, 0.999, "without_", eps_var, mu_var, "full_")
#run(server_step * 5, 5, 0.4, "with_", eps_var, mu_var, "k10")

C = 5
for E in [1, 5, 10]:
    run(server_step * E, E, 0.999, "with_", eps_var, mu_var, C, "c5_")