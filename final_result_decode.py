import pickle as pkl
import numpy as np
import os
import sys

PATH = "results/driving_results/fast_slow_fast-feas_conf.pkl"
with open(PATH, 'rb') as f:
    data = pkl.load(f)
print(type(data[0]))