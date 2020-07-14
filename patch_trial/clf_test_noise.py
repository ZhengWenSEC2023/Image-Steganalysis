# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:38:04 2020

@author: Lenovo
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

pos_sample_test = np.concatenate(
    (np.squeeze(np.load(r"/mnt/zhengwen/new_trial/test_ori_ori_output_2_noise.npy")),
     np.squeeze(np.load(r"/mnt/zhengwen/new_trial/test_ori_ste_output_2_noise.npy"))), axis=1)

neg_sample_test = np.concatenate(
    (np.squeeze(np.load(r"/mnt/zhengwen/new_trial/test_ste_ori_output_2_noise.npy")),
     np.squeeze(np.load(r"/mnt/zhengwen/new_trial/test_ste_ste_output_2_noise.npy"))), axis=1)

pos_sample_test = pos_sample_test.reshape((pos_sample_test.shape[0], -1))
neg_sample_test = neg_sample_test.reshape((neg_sample_test.shape[0], -1))

label_pos = np.ones(len(pos_sample_test))
label_neg = -np.ones(len(neg_sample_test))

sample_test = np.concatenate((pos_sample_test, neg_sample_test), axis=0)
label_test = np.concatenate((label_pos, label_neg), axis=0)

# idx = np.random.permutation(len(label_test))
# sample_test = sample_test[idx]
# label_test = label_test[idx]

# f = open('/mnt/zhengwen/new_trial/RMclf.pkl', 'rb') # PixelHop3
f = open('/mnt/zhengwen/new_trial/RMclf_p2_noise.pkl', 'rb')
clf = pickle.load(f)
f.close()

print(clf.score(sample_test, label_test))
