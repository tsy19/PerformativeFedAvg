import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import math
import random
from types import prepare_class

# ax.set_xticks([0,10000,20000,30000,40000,50000])
# ax.set_xticklabels(['0','10k','20k','30k','40k','50k'])
def val(record_theta_avg):
  vals = np.abs(record_theta_avg-mu_PS)

  avg_val = np.mean(vals, axis=0)
  std_val = np.std(vals, axis=0)
  return avg_val, std_val
def plott(record_theta_fedavg, mu_PS, E, K, N, label, C):
  interval = 100
  # record_theta_fedavg = record_theta_fedavg[:, ::E]
  steps = np.arange(record_theta_fedavg.shape[1])
  record_theta_fedavg = record_theta_fedavg[:, ::interval]
  steps = steps[::interval]
  steps = steps * C * (K / (N + 1))
  avg_val_fedavg, _ = val(record_theta_fedavg)
  vals = np.abs(record_theta_fedavg-mu_PS)

  avg_val = np.mean(vals, axis=0)
  std_val = np.std(vals, axis=0)

  lb = avg_val - z*std_val/np.sqrt(num_rep)
  ub = avg_val + z*std_val/np.sqrt(num_rep)
  # plt.plot(avg_val_fedsgd, label="DSGD-GD")
  plt.plot(steps, avg_val_fedavg, label=label)
  plt.fill_between(steps, lb, ub, alpha=.2)

mu_PS = 100
num_rep = 5
# record_theta_fedavg_2 = np.load("with_E5N25K2.npy")
# record_theta_fedavg_5_with = np.load("with_E5N25K5.npy")
# record_theta_fedavg_15 = np.load("with_E5N25K15.npy")
# record_theta_fedavg_20 = np.load("with_E5N25K20.npy")
# record_theta_fedavg_25 = np.load("with_E5N25K25.npy")

# record_theta_fedavg_5_without = np.load("without_E5N25K5.npy")
C = 10
record_theta_fedavg_7 = np.load("with_E1N25K7straggler_.npy")
record_theta_fedavg_12 = np.load("with_E1N25K12straggler_.npy")
record_theta_fedavg_17 = np.load("with_E1N25K17straggler_.npy")
record_theta_fedavg_25 = np.load("with_E1N25K25straggler_.npy")
fig = plt.figure(figsize=(8,6))
ax = plt.gca()
z = 1
# # plt.plot(0, avg_val[0], marker = '*', ms=15)
# plott(record_theta_fedavg_2, mu_PS, 5, 2, 25, "K=2")
# plott(record_theta_fedavg_5_without, mu_PS, 5, 5, 25, "without replacement")
# plott(record_theta_fedavg_5_with, mu_PS, 5, 5, 25, "with replacement")
# plott(record_theta_fedavg_15, mu_PS, 5, 15, 25, "K=15")
# plott(record_theta_fedavg_20, mu_PS, 5, 20, 25, "K=20")
# plott(record_theta_fedavg_25, mu_PS, 5, 25, 25, "K=25")
plott(record_theta_fedavg_7, mu_PS, 1, 7, 25, "K=7", C)
plott(record_theta_fedavg_12, mu_PS, 1, 12, 25, "K=12", C)
plott(record_theta_fedavg_17, mu_PS, 1, 17, 25, "K=17", C)
plott(record_theta_fedavg_25, mu_PS, 1, 25, 25, "K=25", C)
# plott(record_theta_fedavg_1_25, mu_PS, 1, 25, 25)
# plott(record_theta_fedavg_5_25, mu_PS, 1, 25, 25)
# plott(record_theta_fedavg_20_25, mu_PS, 20, 25, 25)
# plott()

plt.xlabel('Local Iterations',fontsize = 15)
plt.ylabel('Distance to the stable point',fontsize = 15)
plt.legend(fontsize = 15, loc = 'upper right')
plt.xlim([0, 600000])
plt.yscale('log')
# plt.xlim([0, 100000])
plt.grid()
# plt.show()
# plt.savefig("with_without_replacement_E5N25K5.pdf")
plt.savefig("ImpactofE_full.pdf")