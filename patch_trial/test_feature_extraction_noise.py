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

ori_noise_map = np.load(r"/mnt/zhengwen/new_trial/ori_noise_map_test.npy")
steg_noise_map = np.load(r"/mnt/zhengwen/new_trial/steg_noise_map_test.npy")

original_path = r'/mnt/zhengwen/new_trial/test_BOSS_resize'
stego_path = r'/mnt/zhengwen/new_trial/test_BOSS_S_UNIWARD_05'

image_names = os.listdir(original_path)
image_names.sort(key=lambda x: int(x[:-4]))

original_vectors, stego_vectors = [], []

for k in range(len(ori_noise_map)):
    ori_noise = ori_noise_map[k][:, :, 0]
    stego_noise = steg_noise_map[k][:, :, 0]
    ori_image = cv2.imread(os.path.join(original_path, image_names[k]), 0)
    stego_image = cv2.imread(os.path.join(stego_path, image_names[k]), 0)
    ori_block = skimage.util.view_as_blocks(ori_noise, (32, 32))
    stego_block = skimage.util.view_as_blocks(stego_noise, (32, 32))
    ori_block_img = skimage.util.view_as_blocks(ori_image, (32, 32))
    stego_block_img = skimage.util.view_as_blocks(stego_image, (32, 32))
    total_different = np.sum(ori_image != stego_image)
    for i in range(8):
        for j in range(8):
            cur_ori_block = ori_block[i, j]
            cur_stego_block = stego_block[i, j]
            cur_ori_block_img = ori_block_img[i, j]
            cur_stego_block_img = stego_block_img[i, j]
            cur_diff_block = np.sum(cur_ori_block_img != cur_stego_block_img)
            if cur_diff_block > total_different / 8 / 8:
                original_vectors.append(cur_ori_block.copy())
                stego_vectors.append(cur_stego_block.copy())


original_vectors = np.array(original_vectors).astype("double") / 2
stego_vectors = np.array(stego_vectors).astype('double') / 2

print(original_vectors.shape)
print(stego_vectors.shape)

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

f = open('/mnt/zhengwen/new_trial/p2_original_noise.pkl', 'rb')
p2_original = pickle.load(f)
f.close()

f = open('/mnt/zhengwen/new_trial/p2_stego_noise.pkl', 'rb')
p2_stego = pickle.load(f)
f.close()

test_ori_ori_output = p2_original.transform(original_vectors[:, :, :, None])  # ori_features into ori pixelhop
test_ste_ori_output = p2_original.transform(stego_vectors[:, :, :, None])  # ste_features into ori pixelhop
test_ori_ste_output = p2_stego.transform(original_vectors[:, :, :, None])  # ori_features into ste pixelhop
test_ste_ste_output = p2_stego.transform(stego_vectors[:, :, :, None])  # ste_features into ste pixelhop

np.save(r"/mnt/zhengwen/new_trial/test_ori_ori_output_1_noise.npy", test_ori_ori_output[0])
np.save(r"/mnt/zhengwen/new_trial/test_ori_ori_output_2_noise.npy", test_ori_ori_output[1])
np.save(r"/mnt/zhengwen/new_trial/test_ori_ori_output_3_noise.npy", test_ori_ori_output[2])
np.save(r"/mnt/zhengwen/new_trial/test_ori_ste_output_1_noise.npy", test_ori_ste_output[0])
np.save(r"/mnt/zhengwen/new_trial/test_ori_ste_output_2_noise.npy", test_ori_ste_output[1])
np.save(r"/mnt/zhengwen/new_trial/test_ori_ste_output_3_noise.npy", test_ori_ste_output[2])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ori_output_1_noise.npy", test_ste_ori_output[0])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ori_output_2_noise.npy", test_ste_ori_output[1])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ori_output_3_noise.npy", test_ste_ori_output[2])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ste_output_1_noise.npy", test_ste_ste_output[0])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ste_output_2_noise.npy", test_ste_ste_output[1])
np.save(r"/mnt/zhengwen/new_trial/test_ste_ste_output_3_noise.npy", test_ste_ste_output[2])

