# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:06:59 2020

@author: Lenovo
"""

import cv2
import os
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

import numpy as np
from skimage.util import view_as_windows
import pickle
from skimage.measure import block_reduce
from final_pixel_pipeline.pixelhop2 import Pixelhop2


original_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
stego_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'

image_names = os.listdir(original_path)
image_names.sort(key=lambda x: int(x[:-4]))

original_vectors, stego_vectors = [], []
position = []
NUM_TRAIN = 2000
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
    if count == NUM_TRAIN:
        break



np.save("/mnt/zhengwen/new_trial/original_vectors.npy", original_vectors)
np.save("/mnt/zhengwen/new_trial/stego_vectors.npy", stego_vectors)

original_vectors = np.array(original_vectors).astype("double") / 255
stego_vectors = np.array(stego_vectors).astype('double') / 255

np.save("/mnt/zhengwen/new_trial/original_vectors_normalized.npy", original_vectors)
np.save("/mnt/zhengwen/new_trial/stego_vectors_normalized.npy", stego_vectors)

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
SaabArgs = [{'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None},
            {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None},
            {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None}]
shrinkArgs = [{'func': Shrink, 'win': 3},
              {'func': Shrink, 'win': 3},
              {'func': Shrink, 'win': 3}]
concatArg = {'func': Concat}

train_patches = np.concatenate((original_vectors, stego_vectors), axis=0)
train_labels = np.concatenate((0 * np.ones(len(original_vectors)), 1 * np.ones(len(stego_vectors))), axis=0)
idx = np.random.permutation(len(train_patches))
train_patches = train_patches[idx]
train_labels = train_labels[idx]

p2_original = Pixelhop2(depth=2, TH1=0.005, TH2=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
p2_original.fit(train_patches[:, :, :, None])

f = open('/mnt/zhengwen/new_trial/p2_original_week8_HONGSHUO', 'wb')
pickle.dump(p2_original, f)
f.close()

train_features = p2_original.transform(train_patches[:, :, :, None])

pixel_hop_2 = train_features[1]
n_channels = pixel_hop_2.shape[-1]
N_train = pixel_hop_2.shape[0]

params = {
    'min_child_weight': [1, 2, 4, 5, 6, 7, 8, 10, 11, 14, 15, 17, 21],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate': [0.1, 0.01, 0.2]
}

folds = 4
param_comb = 20
clf_list = []
for i in range(n_channels):
    print(i)
    train_flat = pixel_hop_2[:, :, :, i].reshape(N_train, -1)
    # XGB
    xgb = XGBClassifier()
    # K-Fold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
                                       n_jobs=-1, cv=skf.split(train_flat, train_labels), random_state=1001)
    random_search.fit(train_flat, train_labels)
    clf_list.append(random_search.best_estimator_)
f = open("clf_list_week8_HONGSHUO.pkl", "wb")
pickle.dump(clf_list, f)
f.close()

train_pred_prob = []
test_pred_prob = []
for i in range(n_channels):
    train_flat = pixel_hop_2[:, :, :, i].reshape(N_train, -1)
    # test_flat = test_feature[:, :, :, i].reshape(N_test, -1)
    clf = clf_list[i]
    train_pred_prob.append(clf.predict_proba(train_flat)[:, 1])
    # test_pred_prob.append(clf.predict_proba(test_flat)[:, 1])


train_pred_prob = np.array(train_pred_prob).transpose()
test_pred_prob = np.array(test_pred_prob).transpose()

log_clf = LogisticRegression(class_weight='balanced', n_jobs=-1, max_iter=1000)
log_clf.fit(train_pred_prob, train_labels)
log_clf.score(train_pred_prob, train_labels)  # 95.801

f = open("final_LOG_week8_HONGSHUO.pkl", "wb")
pickle.dump(log_clf, f)
f.close()
