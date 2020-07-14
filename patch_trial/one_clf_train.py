# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:36:25 2020

@author: Lenovo
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

pos_sample_train = np.squeeze(np.load(r'/mnt/zhengwen/new_trial/train_ori_sing_output_2.npy'))
neg_sample_train = np.squeeze(np.load(r"/mnt/zhengwen/new_trial/train_ste_sing_output_2.npy"))

pos_sample_train = pos_sample_train.reshape((pos_sample_train.shape[0], -1))
neg_sample_train = neg_sample_train.reshape((neg_sample_train.shape[0], -1))


label_pos = np.ones(len(pos_sample_train))
label_neg = -np.ones(len(neg_sample_train))

sample_train = np.concatenate((pos_sample_train, neg_sample_train), axis=0)
label_train = np.concatenate((label_pos, label_neg), axis=0)

idx = np.random.permutation(len(label_train))
sample_train = sample_train[idx]
label_train = label_train[idx]

clf = RandomForestClassifier(random_state=0, max_depth=200, n_estimators=200)
clf.fit(sample_train, label_train)
print("Training score is:", clf.score(sample_train, label_train))

# f = open('/mnt/zhengwen/new_trial/RMclf_single.pkl', 'wb')    #### PixelHop3 features

f = open('/mnt/zhengwen/new_trial/RMclf_single_p2.pkl', 'wb')
pickle.dump(clf, f)
f.close()