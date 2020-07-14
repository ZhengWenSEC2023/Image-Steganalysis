# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:06:59 2020

@author: Lenovo
"""

from patch_trial.pixelhop2 import Pixelhop2
import numpy as np
from skimage.util import view_as_windows
import pickle
from skimage.measure import block_reduce

original_vectors = np.load("/mnt/zhengwen/new_trial/original_vectors.npy")
stego_vectors = np.load("/mnt/zhengwen/new_trial/stego_vectors.npy")

vectors = np.concatenate((original_vectors, stego_vectors), axis=0)


def ShrinkAvg(X, shrinkArg):
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    X = view_as_windows(X, (1, win, win, 1), (1, stride, stride, 1))
    X_avg_pool = block_reduce(X.reshape(X.shape[0], X.shape[1], X.shape[2], -1), (1, 2, 2, 1), np.average)
    return X_avg_pool


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
shrinkArgs = [{'func': ShrinkAvg, 'win': 5, 'stride': 2},
              {'func': ShrinkAvg, 'win': 5, 'stride': 2},
              {'func': ShrinkAvg, 'win': 5, 'stride': 2}]
concatArg = {'func': Concat}

p2_original = Pixelhop2(depth=3, TH1=0.05, TH2=0.00001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
p2_original.fit(vectors[:, :, :, None])
kernels_original = getKernels(p2_original)

f = open('/mnt/zhengwen/new_trial/p2_single_reverse.pkl', 'wb')
pickle.dump(p2_original, f)
f.close()

f = open('/mnt/zhengwen/new_trial/p2_kernel_single_reverse.pkl', 'wb')
pickle.dump(kernels_original, f)
f.close()

