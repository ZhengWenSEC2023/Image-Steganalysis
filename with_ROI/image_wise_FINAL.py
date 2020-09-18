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

NUM_TRAIN_FINAL = 3000
PERCENTAGE = 0.05
NUM_POINTS = int(PERCENTAGE * 256 * 256)

channels = [3, 7, 14, 20, 22, 23, 25, 26, 27, 33, 40, 46, 47, 48, 51, 52,
            58, 68, 71, 76, 77, 80]
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
f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopUniform_singPH.pkl", 'rb')
p2 = pickle.load(f)
f.close()

f_ori_context = p2.transform(f_ori_train_img)
f_steg_context = p2.transform(f_steg_train_img)

f_ori_context_1 = f_ori_context[0]
f_steg_context_1 = f_steg_context[0]

f_ori_context_1 = f_ori_context_1[:, :, :, channels]
f_steg_context_1 = f_steg_context_1[:, :, :, channels]

ori_c1 = []
steg_c1 = []
for k in range(len(f_ori_context_1)):
    cur_ori_c1 = f_ori_context_1[k]

    cur_ori_rho = f_ori_train_rho[k]
    cur_ori_rho = cur_ori_rho.reshape(-1)

    cur_visit = np.zeros(256 * 256)
    cur_ori_idx = np.argsort(cur_ori_rho, kind="quicksort")
    cur_visit[cur_ori_idx[:NUM_POINTS]] = 1
    cur_visit = cur_visit.reshape((256, 256))

    cur_vector = []

    for i in range(256):
        for j in range(256):
            if cur_visit[i, j] == 1:
                pos_1 = (i, j)
                pos_2 = (i - 1 if i - 1 >= 0 else -(i - 1), j - 1 if j - 1 >= 0 else -(j - 1))
                pos_3 = (i - 1 if i - 1 >= 0 else -(i - 1), j)
                pos_4 = (i - 1 if i - 1 >= 0 else -(i - 1), j + 1 if j + 1 < cur_visit.shape[1] else cur_visit.shape[1] - (j + 1 - cur_visit.shape[1] + 1))
                pos_5 = (i, j - 1 if j - 1 >= 0 else -(j - 1))
                pos_6 = (i, j + 1 if j + 1 < cur_visit.shape[1] else cur_visit.shape[1] - (j + 1 - cur_visit.shape[1] + 1))
                pos_7 = (i + 1 if i + 1 < cur_visit.shape[0] else cur_visit.shape[0] - (i + 1 - cur_visit.shape[0] + 1), j - 1 if j - 1 >= 0 else -(j - 1))
                pos_8 = (i + 1 if i + 1 < cur_visit.shape[0] else cur_visit.shape[0] - (i + 1 - cur_visit.shape[0] + 1), j)
                pos_9 = (i + 1 if i + 1 < cur_visit.shape[0] else cur_visit.shape[0] - (i + 1 - cur_visit.shape[0] + 1), j + 1 if j + 1 < cur_visit.shape[1] else cur_visit.shape[1] - (j + 1 - cur_visit.shape[1] + 1))
                cur_vector.append(
                    np.concatenate(
                        (cur_ori_c1[pos_1], cur_ori_c1[pos_2], cur_ori_c1[pos_3],
                         cur_ori_c1[pos_4], cur_ori_c1[pos_5], cur_ori_c1[pos_6],
                         cur_ori_c1[pos_7], cur_ori_c1[pos_8], cur_ori_c1[pos_9],), axis=0
                    )
                )

    cur_vector = np.array(cur_vector)
    ori_c1.append(cur_vector)
    # Stego
    cur_steg_c1 = f_steg_context_1[k]

    cur_steg_rho = f_steg_train_rho[k]
    cur_steg_rho = cur_steg_rho.reshape(-1)

    cur_visit = np.zeros(256 * 256)
    cur_steg_idx = np.argsort(cur_steg_rho, kind="quicksort")
    cur_visit[cur_steg_idx[:NUM_POINTS]] = 1
    cur_visit = cur_visit.reshape((256, 256))

    cur_vector = []

    for i in range(256):
        for j in range(256):
            if cur_visit[i, j] == 1:
                pos_1 = (i, j)
                pos_2 = (i - 1 if i - 1 >= 0 else -(i - 1), j - 1 if j - 1 >= 0 else -(j - 1))
                pos_3 = (i - 1 if i - 1 >= 0 else -(i - 1), j)
                pos_4 = (i - 1 if i - 1 >= 0 else -(i - 1), j + 1 if j + 1 < cur_visit.shape[1] else cur_visit.shape[1] - (j + 1 - cur_visit.shape[1] + 1))
                pos_5 = (i, j - 1 if j - 1 >= 0 else -(j - 1))
                pos_6 = (i, j + 1 if j + 1 < cur_visit.shape[1] else cur_visit.shape[1] - (j + 1 - cur_visit.shape[1] + 1))
                pos_7 = (i + 1 if i + 1 < cur_visit.shape[0] else cur_visit.shape[0] - (i + 1 - cur_visit.shape[0] + 1), j - 1 if j - 1 >= 0 else -(j - 1))
                pos_8 = (i + 1 if i + 1 < cur_visit.shape[0] else cur_visit.shape[0] - (i + 1 - cur_visit.shape[0] + 1), j)
                pos_9 = (i + 1 if i + 1 < cur_visit.shape[0] else cur_visit.shape[0] - (i + 1 - cur_visit.shape[0] + 1), j + 1 if j + 1 < cur_visit.shape[1] else cur_visit.shape[1] - (j + 1 - cur_visit.shape[1] + 1))
                cur_vector.append(
                    np.concatenate(
                        (cur_steg_c1[pos_1], cur_steg_c1[pos_2], cur_steg_c1[pos_3],
                         cur_steg_c1[pos_4], cur_steg_c1[pos_5], cur_steg_c1[pos_6],
                         cur_steg_c1[pos_7], cur_steg_c1[pos_8], cur_steg_c1[pos_9],), axis=0
                    )
                )
    cur_vector = np.array(cur_vector)
    steg_c1.append(cur_vector)

ori_c1 = np.array(ori_c1)
steg_c1 = np.array(steg_c1)

ori_context = ori_c1.reshape(-1, 198)
steg_context = steg_c1.reshape(-1, 198)

np.save("final_ori_context.npy", ori_context)
np.save("final_steg_context.npy", steg_context)

print("BEGIN CALCULATE PROBABILITY FOR EACH PIXELHOP")

ori_context = np.load("final_ori_context.npy")
steg_context = np.load("final_steg_context.npy")

f = open("single_singPH1_with_selection.pkl", "rb")
xgb = pickle.load(f)
f.close()

ori_feature = xgb.predict_proba(ori_context)[:, 1]
steg_feature = xgb.predict_proba(steg_context)[:, 1]

# ori_feature = xgb.predict(ori_context)
# steg_feature = xgb.predict(steg_context)

ori_pred = ori_feature.reshape((NUM_TRAIN_FINAL, -1))
steg_pred = steg_feature.reshape((NUM_TRAIN_FINAL, -1))

ori_pred_avg = np.sum(ori_pred == 1, axis=1)
steg_pred_avg = np.sum(steg_pred == 1, axis=1)

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
