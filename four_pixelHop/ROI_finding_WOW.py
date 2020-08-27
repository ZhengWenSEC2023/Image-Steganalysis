from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import heapq
import numpy as np

NUM_TRAIN_FINAL = 2000
PERCENTAGE = 0.05

NUM_POINTS = int(PERCENTAGE * 256 * 256)

ori_rho_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize_rho'
steg_rho_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05_rho'

file_names = os.listdir(ori_rho_path)
file_names.sort(key=lambda x: int(x[:-8]))
count = 0
for file_name in file_names:
    ori_rho = loadmat(os.path.join(ori_rho_path, file_name))["rho"]
    original_shape = ori_rho.shape
    ori_rho = ori_rho.reshape(-1)
    ori_idxs = np.argsort(ori_rho, kind="quicksort")
    ori_rho = ori_rho[ori_idxs[:NUM_POINTS]]

