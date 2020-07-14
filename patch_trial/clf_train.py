# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:36:25 2020

@author: Lenovo
"""
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import pickle

pos_sample_train = np.concatenate(
    (np.squeeze(np.load(r"/mnt/zhengwen/new_trial/train_ori_ori_output_2.npy")),
     np.squeeze(np.load(r"/mnt/zhengwen/new_trial/train_ori_ste_output_2.npy"))), axis=1)

neg_sample_train = np.concatenate(
    (np.squeeze(np.load(r"/mnt/zhengwen/new_trial/train_ste_ori_output_2.npy")),
     np.squeeze(np.load(r"/mnt/zhengwen/new_trial/train_ste_ste_output_2.npy"))), axis=1)

pos_sample_train = pos_sample_train.reshape((pos_sample_train.shape[0], -1))
neg_sample_train = neg_sample_train.reshape((neg_sample_train.shape[0], -1))

label_pos = np.ones(len(pos_sample_train))
label_neg = -np.ones(len(neg_sample_train))

sample_train = np.concatenate((pos_sample_train, neg_sample_train), axis=0)
label_train = np.concatenate((label_pos, label_neg), axis=0)

idx = np.random.permutation(len(label_train))
sample_train = sample_train[idx]
label_train = label_train[idx]

clf = RandomForestClassifier(random_state=0)
clf.fit(sample_train, label_train)

# f = open('/mnt/zhengwen/new_trial/RMclf.pkl', 'wb') PixelHop 3
f = open('/mnt/zhengwen/new_trial/RMclf_p2.pkl', 'wb')
pickle.dump(clf, f)
f.close()
