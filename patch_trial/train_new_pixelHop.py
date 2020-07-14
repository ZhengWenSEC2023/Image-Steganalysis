# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:06:59 2020

@author: Lenovo
"""

import cv2
import os
import skimage

from patch_trial.pixelhop2 import Pixelhop2
import numpy as np
from skimage.util import view_as_windows
import pickle
from skimage.measure import block_reduce


def sobel(gray_image):
    def gradNorm(grad):
        return (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
    
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 1, 1, cv2.BORDER_DEFAULT)
    m, n = gray_image.shape[0], gray_image.shape[1]
    panel_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    panel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # panel_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    # panel_Y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
    X = cv2.filter2D(gray_image, -1, panel_X)
    Y = cv2.filter2D(gray_image, -1, panel_Y)
    grad = np.sqrt((X ** 2) + (Y ** 2))
    return gradNorm(grad).astype("double")


original_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
original_sobel_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize_edges'
stego_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'
stego_sobel_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05_edges'

image_names = os.listdir(original_path)
image_names.sort(key=lambda x:int(x[:-4]))

original_vectors, stego_vectors = [], []

NUM_TRAIN = 2000
count = 0
for image_name in image_names:
    ori_image = cv2.imread( os.path.join(original_path, image_name), 0 )
    stego_image = cv2.imread( os.path.join(stego_path, image_name), 0 )
    ori_grad = sobel(ori_image)
    stego_grad = sobel(stego_image)        
    ori_block = skimage.util.view_as_blocks(ori_image, (32, 32))
    stego_block = skimage.util.view_as_blocks(stego_image, (32, 32))
    total_different = np.sum(ori_image != stego_image)
    for i in range(8):
        for j in range(8):
            cur_ori_block = ori_block[i, j]
            cur_stego_block = stego_block[i, j]
            cur_diff_block = np.sum(cur_ori_block != cur_stego_block)
            if cur_diff_block > total_different / 8 / 8:
                original_vectors.append(cur_ori_block.copy())
                stego_vectors.append(cur_stego_block.copy())
    count += 1
    if count == NUM_TRAIN:
        break

original_vectors = np.array(original_vectors).astype("double") / 255
stego_vectors = np.array(stego_vectors).astype('double') / 255

np.save("/mnt/zhengwen/new_trial/original_vectors.npy", original_vectors)
np.save("/mnt/zhengwen/new_trial/stego_vectors.npy", stego_vectors)

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

def getKernels(pixelHop):
    pars = pixelHop.par
    layer_list = list(pars.keys())
    layer_list.sort(key=lambda x:int(x[-1]))
    total_kernels = []
    for key in layer_list:
        current_kernel = []
        for each_saab in pars[key]:
            current_kernel.append(each_saab.Kernels)
        total_kernels.append(current_kernel)
    return total_kernels
        

# PixelHop++
SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':False, 'batch':None, 'cw':False}, 
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':False, 'batch':None, 'cw':True},
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':False, 'batch':None, 'cw':True}]
shrinkArgs = [{'func':ShrinkAvg, 'win':5, 'stride':1},
             {'func': ShrinkAvg, 'win':5, 'stride':1},
             {'func': ShrinkAvg, 'win':5, 'stride':1}]
concatArg = {'func':Concat}

p2_original = Pixelhop2(depth=3, TH1=0.05, TH2=0.00001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
p2_original.fit(original_vectors[:, :, :, None])
kernels_original = getKernels(p2_original)

p2_stego = Pixelhop2(depth=3, TH1=0.05, TH2=0.00001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
p2_stego.fit(stego_vectors[:, :, :, None])
kernels_stego = getKernels(p2_stego)

f = open('/mnt/zhengwen/new_trial/p2_original_reverse.pkl', 'wb')
pickle.dump(p2_original, f)
f.close()

f = open('/mnt/zhengwen/new_trial/kernels_original_reverse.pkl', 'wb')
pickle.dump(kernels_original, f)
f.close()

f = open('/mnt/zhengwen/new_trial/p2_stego_reverse.pkl', 'wb')
pickle.dump(p2_stego, f)
f.close()

f = open('/mnt/zhengwen/new_trial/kernels_stego_reverse.pkl', 'wb')
pickle.dump(kernels_stego, f)
f.close()