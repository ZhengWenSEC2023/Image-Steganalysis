from patch_trial.pixelhop2 import Pixelhop2
from patch_trial.cross_entropy import Cross_Entropy
from patch_trial.lag import LAG
from tensorflow.keras.datasets import cifar10
import numpy as np
from skimage.util import view_as_windows
import pickle
from skimage.measure import block_reduce
from patch_trial.llsr import LLSR
import time
import multiprocessing as mp
from joblib import Parallel, delayed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier

np.random.seed(20)	
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train_later = np.reshape(x_train.copy(), (50000, -1))
y_train_later = y_train.copy()

x_train = np.reshape(x_train, (50000, -1))
num_train = 2000   # number of each training class

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

# train
start_time = time.time()

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
shrinkArgs = [{'func':ShrinkAvg, 'win':7, 'stride':1},
             {'func': ShrinkAvg, 'win':5, 'stride':1},
             {'func': ShrinkAvg, 'win':3, 'stride':1}]
concatArg = {'func':Concat}

p2_avg = Pixelhop2(depth=3, TH1=0.001, TH2=0.0001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
p2_avg.fit(x_train)
# output_train_avg = p2_avg.transform(x_train)

SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None, 'cw':False},
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw':True},
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw':True}]
shrinkArgs = [{'func':ShrinkMax, 'win':7, 'stride':1},
             {'func': ShrinkMax, 'win':5, 'stride':1},
             {'func': ShrinkMax, 'win':3, 'stride':1}]
concatArg = {'func':Concat}

p2_max = Pixelhop2(depth=3, TH1=0.001, TH2=0.0001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
p2_max.fit(x_train)
# output_train_max = p2_max.transform(x_train)

f = open('pixelHop2_AVG_double.pkl', 'wb')
pickle.dump(p2_avg, f)
f.close()

f = open('pixelHop2_MAX_double.pkl', 'wb')
pickle.dump(p2_max, f)
f.close()

print()
print('Pixel Hop saved!')
print()

# f = open('pixelHop2.pkl', 'rb') 
# p2 = pickle.load(f)
# f.close()

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

x_train_later_ori = x_train_later.copy()
y_train_later_ori = y_train_later.copy()

image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range = 0.05, 
            width_shift_range=0.07,
            height_shift_range=0.07,
            horizontal_flip=False,
            vertical_flip=False, 
            data_format="channels_last",
            zca_whitening=False)

train_later_size = x_train_later.shape[0]
augment_size = int(3/10 * train_later_size)

if augment_size != 0:

    image_generator.fit(x_train_later, augment=True, seed=23)
    randidx = np.random.randint(train_later_size, size=augment_size)

    x_augmented = x_train_later[randidx].copy()
    y_augmented = y_train_later[randidx].copy()

    x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                        batch_size=augment_size, shuffle=False, seed=23).next()[0]

    x_train_later = np.concatenate((x_train_later, x_augmented))
    y_train_later = np.concatenate((y_train_later, y_augmented))


output_train_avg = p2_avg.transform(x_train_later)
output_train_max = p2_max.transform(x_train_later)


Ns = 1500
feature_set = []

class Cal_ce:
    def __init__(self, output_train_sub):
        self.ce = Cross_Entropy(num_class=10, num_bin=5)    

    def ce_cal(self, each_out, label):
        each_out = each_out.reshape(-1,1)
        return self.ce.compute(each_out, label)
    

for each in output_train_avg:
    each = each.reshape(each.shape[0], -1)
    ce = Cross_Entropy(num_class=10, num_bin=5)    
    cal_ce = Cal_ce(ce)
    feat_ce = Parallel(n_jobs=mp.cpu_count(), backend='multiprocessing')(delayed(cal_ce.ce_cal)(each[:,k], y_train_later) for k in range(each.shape[-1]))
    feat_ce = np.array(feat_ce)
    feature_set.append(np.argpartition(feat_ce, np.min( (Ns, each.shape[-1] - 1) ))[:np.min( (Ns, each.shape[-1]) )])
    
for each in output_train_max:
    each = each.reshape(each.shape[0], -1)
    ce = Cross_Entropy(num_class=10, num_bin=5)
    cal_ce = Cal_ce(ce)
    feat_ce = Parallel(n_jobs=mp.cpu_count(), backend='multiprocessing')(delayed(cal_ce.ce_cal)(each[:,k], y_train_later) for k in range(each.shape[-1]))
    feat_ce = np.array(feat_ce)
    feature_set.append(np.argpartition(feat_ce, np.min( (Ns, each.shape[-1] - 1) ))[:np.min( (Ns, each.shape[-1]) )])


f = open('feature_set_avg_max_double.pkl', 'wb')
pickle.dump(feature_set, f)
f.close()

print()
print('feature_set saved!')
print()

# # f = open('feature_set.pkl', 'rb') 
# # feature_set = pickle.load(f)
# # f.close()

lag_rep = [
    LAG(encode='distance', num_clusters=[7] * 10, alpha=10, learner=LLSR(onehot=False)),
    LAG(encode='distance', num_clusters=[6] * 10, alpha=10, learner=LLSR(onehot=False)),
    LAG(encode='distance', num_clusters=[5] * 10, alpha=10, learner=LLSR(onehot=False)),
    LAG(encode='distance', num_clusters=[7] * 10, alpha=10, learner=LLSR(onehot=False)),
    LAG(encode='distance', num_clusters=[6] * 10, alpha=10, learner=LLSR(onehot=False)),
    LAG(encode='distance', num_clusters=[5] * 10, alpha=10, learner=LLSR(onehot=False)),
    ]

x_train_trans = []
# single
# for i in range(len(lag_rep)):
#     output_i_train = output_train_avg[i].reshape(output_train_avg[i].shape[0], -1)[:, feature_set[i]]
#     lag_rep[i].fit(output_i_train, y_train_later)
#     if isinstance(x_train_trans, list):
#         x_train_trans = lag_rep[i].transform(output_i_train)
#     else:
#         x_train_trans = np.concatenate((x_train_trans, lag_rep[i].transform(output_i_train)), axis=1)

# double
for i in range(len(lag_rep) // 2):
    output_i_train = output_train_avg[i].reshape(output_train_avg[i].shape[0], -1)[:, feature_set[i]]
    lag_rep[i].fit(output_i_train, y_train_later)
    if isinstance(x_train_trans, list):
        x_train_trans = lag_rep[i].transform(output_i_train)
    else:
        x_train_trans = np.concatenate((x_train_trans, lag_rep[i].transform(output_i_train)), axis=1)

for i in range(len(lag_rep) // 2, len(lag_rep)):
    output_i_train = output_train_max[i - len(lag_rep) // 2].reshape(output_train_max[i - len(lag_rep) // 2].shape[0], -1)[:, feature_set[i]]
    lag_rep[i].fit(output_i_train, y_train_later)
    if isinstance(x_train_trans, list):
        x_train_trans = lag_rep[i].transform(output_i_train)
    else:
        x_train_trans = np.concatenate((x_train_trans, lag_rep[i].transform(output_i_train)), axis=1)

f = open('lag_rep_double.pkl', 'wb')
pickle.dump(lag_rep, f)
f.close()

print()
print('lag_rep saved!')
print()



# # f = open('lag_rep.pkl', 'rb') 
# # lag_rep = pickle.load(f)
# # f.close()

# svmclf = svm.SVC(C = 30, kernel='rbf', gamma=1) # 50000
# # svmclf = svm.SVC(C = 0.5, kernel='rbf', gamma=5) # 12500
# # svmclf = svm.SVC(C = 0.2, kernel='rbf', gamma=1.3) # 6250
# # svmclf = svm.SVC(C = 0.005, kernel='rbf', gamma=0.8) # 3120
# # svmclf = svm.SVC(C = 0.005, kernel='rbf', gamma=0.8) # 1560
# svmclf.fit(x_train_trans, np.squeeze(y_train_later))



# f = open('svm.pkl', 'wb') 
# pickle.dump(svmclf, f)
# f.close()

# print()
# print('svm saved!')
# print()

# # f = open('rf.pkl', 'rb') 
# # rf = pickle.load(f)
# # f.close()

end_time = time.time()
print('training time is ', end_time - start_time)

# test
output_test_avg = p2_avg.transform(x_test)
output_test_max = p2_max.transform(x_test)

x_test_trans = []

# single
# for i in range(len(lag_rep)):
#     output_i_test = output_test_avg[i].reshape(output_test_avg[i].shape[0], -1)[:, feature_set[i]]
#     if isinstance(x_test_trans, list):
#         x_test_trans = lag_rep[i].transform(output_i_test)
#     else:
#         x_test_trans = np.concatenate((x_test_trans, lag_rep[i].transform(output_i_test)), axis=1)

# double
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

f = open('x_test_trans_double.pkl', 'wb')
pickle.dump(x_test_trans, f)
f.close()

output_train_avg = p2_avg.transform(x_train_later_ori)
output_train_max = p2_max.transform(x_train_later_ori)

x_train_trans = []

# single
# for i in range(len(lag_rep)):
#     output_i_train = output_train_avg[i].reshape(output_train_avg[i].shape[0], -1)[:, feature_set[i]]
#     if isinstance(x_train_trans, list):
#         x_train_trans = lag_rep[i].transform(output_i_train)
#     else:
#         x_train_trans = np.concatenate((x_train_trans, lag_rep[i].transform(output_i_train)), axis=1)

# double
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

f = open('x_train_trans_double.pkl', 'wb')
pickle.dump(x_train_trans, f)
f.close()

# svmclf = svm.SVC(C = 30, kernel='rbf', gamma=1)
# svmclf.fit(x_train_trans, np.squeeze(y_train_later_ori))
# score = svmclf.score(x_test_trans, np.squeeze(y_test))
# print('Predicted score =', score)
# score_tr = svmclf.score(x_train_trans, np.squeeze(y_train_later_ori))
# print('Train score =', score_tr)

rfclf = RandomForestClassifier(n_estimators=64, random_state=20)
rfclf.fit(x_train_trans, np.squeeze(y_train_later))
score = rfclf.score(x_test_trans, np.squeeze(y_test))
print('Predicted score =', score)
score_tr = rfclf.score(x_train_trans, np.squeeze(y_train_later))
print('Train score =', score_tr)