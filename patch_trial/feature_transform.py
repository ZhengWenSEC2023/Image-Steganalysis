# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:27:55 2020

@author: Lenovo
"""

import numpy as np
from skimage.util import view_as_windows
import pickle
from skimage.measure import block_reduce


def ShrinkAvg(X, shrinkArg):
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    X = view_as_windows(X, (1,win,win,1), (1,stride,stride,1))
    X_avg_pool = block_reduce(X.reshape(X.shape[0], X.shape[1], X.shape[2], -1), (1, 2, 2, 1), np.average)
    return X_avg_pool

def ShrinkMax(X, shrinkArg):
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    X = view_as_windows(X, (1,win,win,1), (1,stride,stride,1))
    X_max_pool = block_reduce(X.reshape(X.shape[0], X.shape[1], X.shape[2], -1), (1, 2, 2, 1), np.max)
    return X_max_pool

def Concat(X, concatArg):
    return X

f = open(r'/mnt/zhengwen/new_trial/p2_original_reverse.pkl', 'rb')
p2_original = pickle.load(f)
f.close()


f = open(r'/mnt/zhengwen/new_trial/p2_stego_reverse.pkl', 'rb')
p2_stego = pickle.load(f)
f.close()

original_vectors = np.load(r"/mnt/zhengwen/new_trial/original_vectors.npy")
stego_vectors = np.load(r"/mnt/zhengwen/new_trial/stego_vectors.npy")

train_ori_ori_output = p2_original.transform(original_vectors[:, :, :, None]) # ori_features into ori pixelhop
train_ste_ori_output = p2_original.transform(stego_vectors[:, :, :, None]) # ste_features into ori pixelhop
train_ori_ste_output = p2_stego.transform(original_vectors[:, :, :, None]) # ori_features into ste pixelhop
train_ste_ste_output = p2_stego.transform(stego_vectors[:, :, :, None]) # ste_features into ste pixelhop

np.save(r"/mnt/zhengwen/new_trial/train_ori_ori_output_1.npy", train_ori_ori_output[0])
np.save(r"/mnt/zhengwen/new_trial/train_ori_ori_output_2.npy", train_ori_ori_output[1])
np.save(r"/mnt/zhengwen/new_trial/train_ori_ori_output_3.npy", train_ori_ori_output[2])
np.save(r"/mnt/zhengwen/new_trial/train_ori_ste_output_1.npy", train_ori_ste_output[0])
np.save(r"/mnt/zhengwen/new_trial/train_ori_ste_output_2.npy", train_ori_ste_output[1])
np.save(r"/mnt/zhengwen/new_trial/train_ori_ste_output_3.npy", train_ori_ste_output[2])
np.save(r"/mnt/zhengwen/new_trial/train_ste_ori_output_1.npy", train_ste_ori_output[0])
np.save(r"/mnt/zhengwen/new_trial/train_ste_ori_output_2.npy", train_ste_ori_output[1])
np.save(r"/mnt/zhengwen/new_trial/train_ste_ori_output_3.npy", train_ste_ori_output[2])
np.save(r"/mnt/zhengwen/new_trial/train_ste_ste_output_1.npy", train_ste_ste_output[0])
np.save(r"/mnt/zhengwen/new_trial/train_ste_ste_output_2.npy", train_ste_ste_output[1])
np.save(r"/mnt/zhengwen/new_trial/train_ste_ste_output_3.npy", train_ste_ste_output[2])

