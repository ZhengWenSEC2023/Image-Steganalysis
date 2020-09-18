from scipy.io import loadmat
import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

#############################################################################################
# TEST SETP: FOR EACH DIFFERENT SET, USE THE TRAINED PIXELHOP, XGBOOST, GIVE THE PERCENTAGE #
#                        CHANGE THE PATH TO TEST DIFFERENT IMAGE SET                        #
#############################################################################################

CHANNEL_RANGE = [0, 15, 15 + 22, 15 + 22 + 28, 15 + 22 + 28 + 39]
NUM_TRAIN_FINAL = 3000
PERCENTAGE = 0.05
NUM_POINTS = int(PERCENTAGE * 256 * 256)

# for rho_file_name in rho_file_names:
#     ori_rho = loadmat(os.path.join(ori_rho_path, rho_file_name))["rho"]
#     original_shape = ori_rho.shape
#     ori_rho = ori_rho.reshape(-1)
#     ori_idxs = np.argsort(ori_rho, kind="quicksort")
#     ori_rho = ori_rho[ori_idxs[:NUM_POINTS]]

def Shrink(X, shrinkArg, max_pooling=True, padding=True):
    if max_pooling:
        X = block_reduce(X, (1, 2, 2, 1), np.max)
    win = shrinkArg['win']
    if padding:
        new_X = []
        for each in X:
            each = cv2.copyMakeBorder(each, win // 2, win // 2, win // 2, win // 2, cv2.BORDER_REFLECT)
            new_X.append(each)
        new_X = np.array(new_X)[:, :, :, None]
    else:
        new_X = X
    X = view_as_windows(new_X, (1, win, win, 1))
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
    return X


# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X

def context_resize(context):
    context = np.moveaxis(context, -1, 1)
    new_context = []
    for i in range(context.shape[0]):
        for j in range(context.shape[1]):
            new_context.append(cv2.resize(context[i, j], (256, 256), interpolation=cv2.INTER_NEAREST))
    new_context = np.array(new_context)
    new_context = np.reshape(new_context, (context.shape[0], context.shape[1], 256, 256))
    new_context = np.moveaxis(new_context, 1, -1)
    return new_context


print("BEGIN READ IMAGE")

f_ori_train_img = []
f_ori_train_rho = []
f_steg_train_img = []
f_steg_train_rho = []
ori_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'
ori_rho_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize_rho'
steg_rho_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05_rho'
img_file_names = os.listdir(ori_img_path)
img_file_names.sort(key=lambda x: int(x[:-4]))
count = 0
for img_file_name in img_file_names:
    rho_file_name = img_file_name + ".mat"
    ori_img = cv2.imread(os.path.join(ori_img_path, img_file_name), 0)[:, :, None]
    ori_rho = loadmat(os.path.join(ori_rho_path, rho_file_name))["rho"]
    steg_img = cv2.imread(os.path.join(steg_img_path, img_file_name), 0)[:, :, None]
    steg_rho = loadmat(os.path.join(steg_rho_path, rho_file_name))["rho"]

    f_ori_train_img.append(ori_img)
    f_ori_train_rho.append(ori_rho)
    f_steg_train_img.append(steg_img)
    f_steg_train_rho.append(steg_rho)
    count += 1
    if count == NUM_TRAIN_FINAL:
        break

f_ori_train_img = np.array(f_ori_train_img)
f_ori_train_rho = np.array(f_ori_train_rho)

f_steg_train_img = np.array(f_steg_train_img)
f_steg_train_rho = np.array(f_steg_train_rho)

# FEATURE EXTRACTION

print("BEGIN FEATURE EXTRACTION")
f = open("PixelHopUniform_4PH.pkl", 'rb')
p2 = pickle.load(f)
f.close()

f_ori_context = p2.transform(f_ori_train_img)
f_steg_context = p2.transform(f_steg_train_img)

f_ori_context_1 = f_ori_context[0]
f_ori_context_2 = f_ori_context[1]
f_ori_context_3 = f_ori_context[2]
f_ori_context_4 = f_ori_context[3]
f_ori_context_2 = context_resize(f_ori_context_2)
f_ori_context_3 = context_resize(f_ori_context_3)
f_ori_context_4 = context_resize(f_ori_context_4)

f_steg_context_1 = f_steg_context[0]
f_steg_context_2 = f_steg_context[1]
f_steg_context_3 = f_steg_context[2]
f_steg_context_4 = f_steg_context[3]
f_steg_context_2 = context_resize(f_steg_context_2)
f_steg_context_3 = context_resize(f_steg_context_3)
f_steg_context_4 = context_resize(f_steg_context_4)

del f_ori_context, f_steg_context

ori_c1 = []
ori_c2 = []
ori_c3 = []
ori_c4 = []
steg_c1 = []
steg_c2 = []
steg_c3 = []
steg_c4 = []

for k in range(len(f_ori_context_1)):
    cur_ori_c1 = f_ori_context_1[k]
    cur_ori_c2 = f_ori_context_2[k]
    cur_ori_c3 = f_ori_context_3[k]
    cur_ori_c4 = f_ori_context_4[k]
    cur_ori_c1 = cur_ori_c1.reshape((256 * 256, -1))
    cur_ori_c2 = cur_ori_c2.reshape((256 * 256, -1))
    cur_ori_c3 = cur_ori_c3.reshape((256 * 256, -1))
    cur_ori_c4 = cur_ori_c4.reshape((256 * 256, -1))
    cur_ori_rho = f_ori_train_rho[k]
    cur_ori_rho = cur_ori_rho.reshape(-1)
    cur_ori_idx = np.argsort(cur_ori_rho, kind="quicksort")
    cur_ori_c1 = cur_ori_c1[cur_ori_idx[:NUM_POINTS]]
    cur_ori_c2 = cur_ori_c2[cur_ori_idx[:NUM_POINTS]]
    cur_ori_c3 = cur_ori_c3[cur_ori_idx[:NUM_POINTS]]
    cur_ori_c4 = cur_ori_c4[cur_ori_idx[:NUM_POINTS]]
    ori_c1.append(cur_ori_c1)
    ori_c2.append(cur_ori_c2)
    ori_c3.append(cur_ori_c3)
    ori_c4.append(cur_ori_c4)

    cur_steg_c1 = f_steg_context_1[k]
    cur_steg_c2 = f_steg_context_2[k]
    cur_steg_c3 = f_steg_context_3[k]
    cur_steg_c4 = f_steg_context_4[k]
    cur_steg_c1 = cur_steg_c1.reshape((256 * 256, -1))
    cur_steg_c2 = cur_steg_c2.reshape((256 * 256, -1))
    cur_steg_c3 = cur_steg_c3.reshape((256 * 256, -1))
    cur_steg_c4 = cur_steg_c4.reshape((256 * 256, -1))
    cur_steg_rho = f_steg_train_rho[k]
    cur_steg_rho = cur_steg_rho.reshape(-1)
    cur_steg_idx = np.argsort(cur_steg_rho, kind="quicksort")
    cur_steg_c1 = cur_steg_c1[cur_steg_idx[:NUM_POINTS]]
    cur_steg_c2 = cur_steg_c2[cur_steg_idx[:NUM_POINTS]]
    cur_steg_c3 = cur_steg_c3[cur_steg_idx[:NUM_POINTS]]
    cur_steg_c4 = cur_steg_c4[cur_steg_idx[:NUM_POINTS]]
    steg_c1.append(cur_steg_c1)
    steg_c2.append(cur_steg_c2)
    steg_c3.append(cur_steg_c3)
    steg_c4.append(cur_steg_c4)

ori_c1 = np.array(ori_c1)
ori_c2 = np.array(ori_c2)
ori_c3 = np.array(ori_c3)
ori_c4 = np.array(ori_c4)
steg_c1 = np.array(steg_c1)
steg_c2 = np.array(steg_c2)
steg_c3 = np.array(steg_c3)
steg_c4 = np.array(steg_c4)

ori_context = np.concatenate(
    (
        ori_c1.reshape((-1, ori_c1.shape[-1])),
        ori_c2.reshape((-1, ori_c2.shape[-1])),
        ori_c3.reshape((-1, ori_c3.shape[-1])),
        ori_c4.reshape((-1, ori_c4.shape[-1])),
    ), axis=-1
)

del ori_c1, ori_c2, ori_c3, ori_c4

steg_context = np.concatenate(
    (
        steg_c1.reshape((-1, steg_c1.shape[-1])),
        steg_c2.reshape((-1, steg_c2.shape[-1])),
        steg_c3.reshape((-1, steg_c3.shape[-1])),
        steg_c4.reshape((-1, steg_c4.shape[-1])),
    ), axis=-1
)

del steg_c1, steg_c2, steg_c3, steg_c4

print("BEGIN CALCULATE PROBABILITY FOR EACH PIXELHOP")

f = open("ensemble_clf_PH4.pkl", "rb")
clf_list = pickle.load(f)
f.close()

ori_prob = []
steg_prob = []
for i in range(1, len(CHANNEL_RANGE)):
    print(i)
    ori_vec = ori_context[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    steg_vec = steg_context[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    clf = clf_list[i - 1]
    ori_prob.append(clf.predict_proba(ori_vec)[:, 1])
    steg_prob.append(clf.predict_proba(steg_vec)[:, 1])

ori_prob = np.array(ori_prob).transpose()
steg_prob = np.array(steg_prob).transpose()

print("BEGIN CALCULATE FINAL LABEL WITH XGBOOST")

f = open("final_XG_BOOST", "rb")
xgboost_final = pickle.load(f)
f.close()

ori_feature = xgboost_final.predict_proba(ori_prob)[:, 1]
steg_feature = xgboost_final.predict_proba(steg_prob)[:, 1]

ori_pred = xgboost_final.predict(ori_prob)
steg_pred = xgboost_final.predict(steg_prob)
ori_pred = ori_pred.reshape((NUM_TRAIN_FINAL, -1))
steg_pred = steg_pred.reshape((NUM_TRAIN_FINAL, -1))

ori_pred_avg = np.mean(ori_pred, axis=1)
steg_pred_avg = np.mean(steg_pred, axis=1)

ori_feature = ori_feature.reshape((NUM_TRAIN_FINAL, -1))
steg_feature = steg_feature.reshape((NUM_TRAIN_FINAL, -1))

ori_label = 0 * np.ones(len(ori_feature))
steg_label = 1 * np.ones(len(steg_feature))

SUB_NUM_POINT = 2000

train_set = np.concatenate((ori_feature[:2700], steg_feature[:2700]), axis=0)
train_label = np.concatenate((ori_label[:2700], steg_label[:2700]), axis=0)
idx = np.random.permutation(len(train_set))
train_set = train_set[idx][:, :SUB_NUM_POINT]
train_label = train_label[idx]

test_set = np.concatenate((ori_feature[2700:], steg_feature[2700:]), axis=0)[:, :SUB_NUM_POINT]
test_label = np.concatenate((ori_label[2700:], steg_label[2700:]), axis=0)

image_wise = SVC(C=2, gamma="scale")
image_wise.fit(train_set, train_label)
train_score = image_wise.score(train_set, train_label)
test_score = image_wise.score(test_set, test_label)
