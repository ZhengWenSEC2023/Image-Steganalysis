# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:52:30 2020

@author: Lenovo
"""

from tensorflow.keras.datasets import cifar10
import numpy as np
from skimage.util import view_as_windows
import pickle
from skimage.measure import block_reduce
import time
from sklearn.ensemble import RandomForestClassifier

start = time.time()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train_later = np.reshape(x_train.copy(), (50000, -1))
y_train_later = y_train.copy()

x_train = np.reshape(x_train, (50000, -1))
num_train = 1000   # number of each training class

if num_train != 5000:
    data_set = []    
    for i in range(10):
        temp = np.concatenate(
            (x_train[np.where(y_train == i)[0], :], y_train[np.where(y_train == i)[0], :]), 
            axis=1)
        temp = temp[:num_train]
        if isinstance(data_set, list):
            data_set = temp
        else:
            data_set = np.concatenate((data_set, temp), axis=0)

    np.random.shuffle(data_set)

    x_train = np.reshape(data_set[:, :-1], (num_train * 10, 32, 32, 3))
    y_train = data_set[:, -1][:, None]
else:
    x_train = np.reshape(x_train, (num_train * 10, 32, 32, 3))
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
num_classes = 10 


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

# PixelHop++
SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None, 'cw':False}, 
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw':True},
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw':True}]
shrinkArgs = [{'func':ShrinkAvg, 'win':5, 'stride':1}, 
             {'func': ShrinkAvg, 'win':5, 'stride':1},
             {'func': ShrinkAvg, 'win':5, 'stride':1}]
concatArg = {'func':Concat}

f = open('pixelHop2_AVG.pkl', 'rb') 
p2_avg = pickle.load(f)
f.close()

f = open('pixelHop2_MAX.pkl', 'rb') 
p2_max = pickle.load(f)
f.close()

factor = 1
num_train_later = int(5000 * factor)  # number of each training class
if num_train_later != 5000:
    data_set = []    
    for i in range(10):
        temp = np.concatenate(
            (x_train_later[np.where(y_train_later == i)[0], :], 
             y_train_later[np.where(y_train_later == i)[0], :]), 
            axis=1)
        np.random.shuffle(temp)
        temp = temp[:num_train_later]
        if isinstance(data_set, list):
            data_set = temp
        else:
            data_set = np.concatenate((data_set, temp), axis=0)

    np.random.shuffle(data_set)

    x_train_later = np.reshape(data_set[:, :-1], (num_train_later * 10, 32, 32, 3))
    y_train_later = data_set[:, -1][:, None]
else:
    x_train_later = np.reshape(x_train_later, (num_train_later * 10, 32, 32, 3))
    
x_train_later = x_train_later.astype('float32')
x_train_later /= 255

output_train_avg = p2_avg.transform(x_train_later)
output_train_max = p2_max.transform(x_train_later)

Ns = 1000

f = open('feature_set_avg_max.pkl', 'rb') 
feature_set = pickle.load(f)
f.close()

f = open('lag_rep.pkl', 'rb') 
lag_rep = pickle.load(f)
f.close()

output_test_avg = p2_avg.transform(x_test)
output_test_max = p2_max.transform(x_test)

x_train_trans = []
for i in range(len(lag_rep) // 2):
    output_i_train = output_train_avg[i].reshape(output_train_avg[i].shape[0], -1)[:, feature_set[i]]
    if isinstance(x_train_trans, list): 
        x_train_trans = lag_rep[i].transform(output_i_train)
    else:
        x_train_trans = np.concatenate((x_train_trans, lag_rep[i].transform(output_i_train)), axis=1)

for i in range(len(lag_rep) // 2, len(lag_rep)):
    output_i_train = output_train_max[i - len(lag_rep) // 2].reshape(output_train_max[i - len(lag_rep) // 2].shape[0], -1)[:, feature_set[i]]
    if isinstance(x_train_trans, list): 
        x_train_trans = lag_rep[i].transform(output_i_train)
    else:
        x_train_trans = np.concatenate((x_train_trans, lag_rep[i].transform(output_i_train)), axis=1)

x_test_trans = []
for i in range(len(lag_rep) // 2):
    output_i_test = output_test_avg[i].reshape(output_test_avg[i].shape[0], -1)[:, feature_set[i]]
    if isinstance(x_test_trans, list): 
        x_test_trans = lag_rep[i].transform(output_i_test)
    else:
        x_test_trans = np.concatenate((x_test_trans, lag_rep[i].transform(output_i_test)), axis=1)

for i in range(len(lag_rep) // 2, len(lag_rep)):
    output_i_test = output_test_max[i - len(lag_rep) // 2].reshape(output_test_max[i - len(lag_rep) // 2].shape[0], -1)[:, feature_set[i]]
    if isinstance(x_test_trans, list): 
        x_test_trans = lag_rep[i].transform(output_i_test)
    else:
        x_test_trans = np.concatenate((x_test_trans, lag_rep[i].transform(output_i_test)), axis=1)


# svmclf = svm.SVC(C = 5, kernel='rbf', gamma=0.5)
# svmclf.fit(x_train_trans, np.squeeze(y_train_later))
# score = svmclf.score(x_test_trans, np.squeeze(y_test))
# print('Predicted score =', score)
# score_tr = svmclf.score(x_train_trans, np.squeeze(y_train_later))
# print('Train score =', score_tr)


rfclf = RandomForestClassifier(max_depth=30, n_estimators=150, max_features=30, random_state=0) # 64.83 New Model
rfclf.fit(x_train_trans, np.squeeze(y_train_later))
score = rfclf.score(x_test_trans, np.squeeze(y_test))
print('Predicted score =', score)
score_tr = rfclf.score(x_train_trans, np.squeeze(y_train_later))
print('Train score =', score_tr)




# svmclf.fit(x_train_trans, np.squeeze(y_train_later))
# score = svmclf.score(x_test_trans, np.squeeze(y_test))
# print('Predicted score =', score)

end_time = time.time()
print(end_time - start)

# score_tr = svmclf.score(x_train_trans, np.squeeze(y_train_later))
# print('Train score =', score_tr)

#########################################
# old structure
#########################################

# rfclf = RandomForestClassifier(max_depth=16, n_estimators=200, max_features=10, random_state=0) # 61.7, Ns = 2000, Augmentation
# rfclf = RandomForestClassifier(max_depth=20, n_estimators=200, max_features=12, random_state=0) # 62.12, Ns = 2000, Augmentation
# rfclf = RandomForestClassifier(max_depth=25, n_estimators=250, max_features=14, random_state=0) # 62.12, Ns = 2000, Augmentation
# rfclf = RandomForestClassifier(max_depth=30, n_estimators=300, max_features=16, random_state=0)

# svmclf = svm.SVC(C = 20, kernel='rbf', gamma=3) # 66.8, Ns = 1000, No augmentation
# svmclf = svm.SVC(C = 20, kernel='rbf', gamma=3) # 64.94, Ns = 2000, Augmentation
# svmclf = svm.SVC(C = 20, kernel='rbf', gamma=5) # 66.06, Ns = 2000, Augmentation
# svmclf = svm.SVC(C = 30, kernel='rbf', gamma=5) # 66.07, Ns = 2000, Augmentation
# svmclf = svm.SVC(C = 50, kernel='rbf', gamma=5) # 66.07�� Ns = 2000, Augmentation