# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:06:59 2020

@author: Lenovo
"""

import cv2
import os

import numpy as np
from skimage.util import view_as_windows
import pickle
from skimage.measure import block_reduce

original_path = r'/mnt/zhengwen/new_trial/test_BOSS_resize'
stego_path = r'/mnt/zhengwen/new_trial/test_BOSS_S_UNIWARD_05'

image_names = os.listdir(original_path)
image_names.sort(key=lambda x: int(x[:-4]))

original_vectors, stego_vectors = [], []
position = []
RATIO = 32 * 32 * 0.2
count = 0
for image_name in image_names:
    ori_image = cv2.imread(os.path.join(original_path, image_name), 0)
    stego_image = cv2.imread(os.path.join(stego_path, image_name), 0)
    ori_window = view_as_windows(ori_image, (32, 32))
    stego_window = view_as_windows(stego_image, (32, 32))
    diff = ori_window.astype("double") - stego_window.astype("double")
    for i in range(ori_window.shape[0]):
        for j in range(ori_window.shape[1]):
            if diff[i, j, 32 // 2, 32 // 2] == 1 and np.sum(abs(diff[i, j])) >= RATIO:
                original_vectors.append(ori_window[i, j])
                stego_vectors.append(stego_window[i, j])
                diff[max(0, i - 32 // 2): min(226, i + 32 // 2), max(0, j - 32 // 2): min(226, j + 32 // 2)] = 0
                position.append((count, i, j))
    count += 1




np.save("/mnt/zhengwen/new_trial/original_vectors_test.npy", original_vectors)
np.save("/mnt/zhengwen/new_trial/stego_vectors_test.npy", stego_vectors)

original_vectors = np.array(original_vectors).astype("double") / 255
stego_vectors = np.array(stego_vectors).astype('double') / 255

np.save("/mnt/zhengwen/new_trial/original_vectors_normalized_test.npy", original_vectors)
np.save("/mnt/zhengwen/new_trial/stego_vectors_normalized_test.npy", stego_vectors)

# def ShrinkAvg(X, shrinkArg):
#     win = shrinkArg['win']
#     stride = shrinkArg['stride']
#     X = view_as_windows(X, (1, win, win, 1), (1, stride, stride, 1))
#     X_avg_pool = block_reduce(X.reshape(X.shape[0], X.shape[1], X.shape[2], -1), (1, 2, 2, 1), np.average)
#     return X_avg_pool
#
#
# def ShrinkMax(X, shrinkArg):
#     win = shrinkArg['win']
#     stride = shrinkArg['stride']
#     X = view_as_windows(X, (1, win, win, 1), (1, stride, stride, 1))
#     X_max_pool = block_reduce(X.reshape(X.shape[0], X.shape[1], X.shape[2], -1), (1, 2, 2, 1), np.max)
#     return X_max_pool

def Shrink(X, shrinkArg, max_pooling=True):
    if max_pooling:
        X = block_reduce(X, (1, 2, 2, 1), np.max)
    win = shrinkArg['win']
    X = view_as_windows(X, (1, win, win, 1))
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
    return X

def Concat(X, concatArg):
    return X

# PixelHop++
f = open('/mnt/zhengwen/new_trial/p2_original_week8_HONGSHUO', 'rb')
p2_original = pickle.load(f)
f.close()

test_patches = np.concatenate((original_vectors, stego_vectors), axis=0)
test_label = np.concatenate((0 * np.ones(len(original_vectors)), 1 * np.ones(len(stego_vectors))), axis=0)

test_features = p2_original.transform(test_patches[:, :, :, None])

pixel_hop_2 = test_features[1]
n_channels = pixel_hop_2.shape[-1]
N_train = pixel_hop_2.shape[0]


f = open("clf_list_week8_HONGSHUO.pkl", "rb")
clf_list = pickle.load(f)
f.close()

test_pred_prob = []
for i in range(n_channels):
    test_flat = pixel_hop_2[:, :, :, i].reshape(N_train, -1)
    clf = clf_list[i]
    test_pred_prob.append(clf.predict_proba(test_flat)[:, 1])

test_pred_prob = np.array(test_pred_prob).transpose()

f = open("final_LOG_week8_HONGSHUO.pkl", "rb")
log_clf = pickle.load(f)
f.close()

print(log_clf.score(test_pred_prob, test_label))
a = 1

