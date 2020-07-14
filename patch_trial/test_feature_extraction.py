# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:04:00 2020

@author: Lenovo
"""

import cv2
import os
import skimage

import numpy as np
from skimage.util import view_as_windows
import pickle
from skimage.measure import block_reduce

test_original_path = r'/mnt/zhengwen/new_trial/test_BOSS_resize'
test_stego_path = r'/mnt/zhengwen/new_trial/test_BOSS_S_UNIWARD_05'

image_names = os.listdir(test_original_path)
image_names.sort(key=lambda x: int(x[:-4]))

original_vectors, stego_vectors = [], []

for image_name in image_names:
    ori_image = cv2.imread(os.path.join(test_original_path, image_name), 0)
    stego_image = cv2.imread(os.path.join(test_stego_path, image_name), 0)
    ori_block = skimage.util.view_as_blocks(ori_image, (32, 32))
    stego_block = skimage.util.view_as_blocks(stego_image, (32, 32))
    original_vectors.append(ori_block.reshape((-1, 32, 32)))
    stego_vectors.append(stego_block.reshape((-1, 32, 32)))

original_vectors = np.concatenate(original_vectors, axis=0).astype("double") / 255
stego_vectors = np.concatenate(stego_vectors, axis=0).astype('double') / 255

np.save(r"/mnt/zhengwen/new_trial/test_original_vectors.npy", original_vectors)
np.save(r"/mnt/zhengwen/new_trial/test_stego_vectors.npy", stego_vectors)


def ShrinkAvg(X, shrinkArg):
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    X = view_as_windows(X, (1, win, win, 1), (1, stride, stride, 1))
    X_avg_pool = block_reduce(X.reshape(X.shape[0], X.shape[1], X.shape[2], -1), (1, 2, 2, 1), np.average)
    return X_avg_pool


def ShrinkMax(X, shrinkArg):
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    X = view_as_windows(X, (1, win, win, 1), (1, stride, stride, 1))
    X_max_pool = block_reduce(X.reshape(X.shape[0], X.shape[1], X.shape[2], -1), (1, 2, 2, 1), np.max)
    return X_max_pool


def Concat(X, concatArg):
    return X


def getKernels(pixelHop):
    pars = pixelHop.par
    layer_list = list(pars.keys())
    layer_list.sort(key=lambda x: int(x[-1]))
    total_kernels = []
    for key in layer_list:
        current_kernel = []
        for each_saab in pars[key]:
            current_kernel.append(each_saab.Kernels)
        total_kernels.append(current_kernel)
    return total_kernels


# PixelHop++
SaabArgs = [{'num_AC_kernels': -1, 'needBias': False, 'useDC': False, 'batch': None, 'cw': False},
            {'num_AC_kernels': -1, 'needBias': True, 'useDC': False, 'batch': None, 'cw': True},
            {'num_AC_kernels': -1, 'needBias': True, 'useDC': False, 'batch': None, 'cw': True}]
shrinkArgs = [{'func': ShrinkMax, 'win': 5, 'stride': 1},
              {'func': ShrinkMax, 'win': 5, 'stride': 1},
              {'func': ShrinkMax, 'win': 5, 'stride': 1}]
concatArg = {'func': Concat}

f = open('/mnt/zhengwen/new_trial/p2_original_reverse.pkl', 'rb')
p2_original = pickle.load(f)
f.close()

f = open('/mnt/zhengwen/new_trial/p2_stego_reverse.pkl', 'rb')
p2_stego = pickle.load(f)
f.close()

test_ori_ori_output = p2_original.transform(original_vectors[:, :, :, None])  # ori_features into ori pixelhop
test_ste_ori_output = p2_original.transform(stego_vectors[:, :, :, None])  # ste_features into ori pixelhop
test_ori_ste_output = p2_stego.transform(original_vectors[:, :, :, None])  # ori_features into ste pixelhop
test_ste_ste_output = p2_stego.transform(stego_vectors[:, :, :, None])  # ste_features into ste pixelhop

np.save(r"/mnt/zhengwen/new_trial/test_ori_ori_output_1.npy", test_ori_ori_output[0])
np.save(r"/mnt/zhengwen/new_trial/test_ori_ori_output_2.npy", test_ori_ori_output[1])
np.save(r"/mnt/zhengwen/new_trial/test_ori_ori_output_3.npy", test_ori_ori_output[2])
np.save(r"/mnt/zhengwen/new_trial/test_ori_ste_output_1.npy", test_ori_ste_output[0])
np.save(r"/mnt/zhengwen/new_trial/test_ori_ste_output_2.npy", test_ori_ste_output[1])
np.save(r"/mnt/zhengwen/new_trial/test_ori_ste_output_3.npy", test_ori_ste_output[2])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ori_output_1.npy", test_ste_ori_output[0])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ori_output_2.npy", test_ste_ori_output[1])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ori_output_3.npy", test_ste_ori_output[2])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ste_output_1.npy", test_ste_ste_output[0])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ste_output_2.npy", test_ste_ste_output[1])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ste_output_3.npy", test_ste_ste_output[2])
