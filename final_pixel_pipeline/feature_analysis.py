import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import os
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

#########################################################
#   STEP 2: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT   #
#     EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT     #
#########################################################

def Shrink(X, shrinkArg, max_pooling=False, padding=True):
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


def diff_sample(diff_maps):
    win = 10
    new_diff_maps = []
    for diff_map in diff_maps:
        visit_map = np.zeros(diff_map.shape)
        for i in range(diff_map.shape[0]):
            for j in range(diff_map.shape[1]):
                if visit_map[i, j] == 0:
                    if diff_map[i, j] != 0:
                        diff_map[max(0, i - win): min(i + win + 1, 256), max(0, j - win): min(j + win + 1, 256)] = 0
                        diff_map[i, j] = 1
                        visit_map[max(0, i - win): min(i + win + 1, 256), max(0, j - win): min(j + win + 1, 256)] = 1
                    else:
                        continue
        new_diff_maps.append(diff_map)
    return np.array(new_diff_maps)


F_TRAIN_NUM_TOTAL = 500
F_TRAIN_NUM_UNCHANGED = 7500000
NUM_VECTOR = 300000

# f = open("PixelHopUniform.pkl", 'rb')  # 3 PixelHop, win: 5, TH1:0.005, TH2:0.005, CH1: 15, CH2: 20, CH3: 25, TRAIN_TOTAL=500
# f = open("PixelHopUniform_singPH.pkl", 'rb')  # 9 * 9
f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopUniform_singPH.pkl", 'rb')
p2 = pickle.load(f)
f.close()

f_ori_train_img = []
f_steg_train_img = []
ori_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'
file_names = os.listdir(ori_img_path)
file_names.sort(key=lambda x: int(x[:-4]))

count = 0
for file_name in file_names:
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)[:, :, None]
    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)[:, :, None]
    f_ori_train_img.append(ori_img)
    f_steg_train_img.append(steg_img)
    count += 1
    if count == F_TRAIN_NUM_TOTAL:
        break

# FEATURE EXTRACTION
f_ori_train_img = np.array(f_ori_train_img)
f_steg_train_img = np.array(f_steg_train_img)
diff_map = np.squeeze(f_ori_train_img.astype("double") - f_steg_train_img.astype("double"))
f_ori_context = p2.transform(f_ori_train_img)
# np.save("conpare_features.npy", f_ori_context)
f_steg_context = p2.transform(f_steg_train_img)

f_ori_context = f_ori_context[0]
f_steg_context = f_steg_context[0]

for i in range(500):
    for j in range(81):
        f_ori_context[i, :, :, j] = (f_ori_context[i, :, :, j] - np.min(f_ori_context[i, :, :, j])) / (np.max(f_ori_context[i, :, :, j]) - np.min(f_ori_context[i, :, :, j]))
        f_steg_context[i, :, :, j] = (f_steg_context[i, :, :, j] - np.min(f_steg_context[i, :, :, j])) / (np.max(f_steg_context[i, :, :, j]) - np.min(f_steg_context[i, :, :, j]))

plt.figure()
for j in range(81):
    plt.subplot(9, 9, j + 1)
    plt.imshow(f_ori_context[0, :, :, j])
    print(f_ori_context[0, :, :, j])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
plt.savefig("Original features.png", dpi=5000)

plt.figure()
for j in range(81):
    plt.subplot(9, 9, j + 1)
    plt.imshow(f_steg_context[0, :, :, j])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
plt.savefig("Stego features.png", dpi=5000)

diff_map = np.squeeze(f_ori_train_img - f_steg_train_img)

plt.figure()
for j in range(81):
    plt.subplot(9, 9, j + 1)
    diff = f_ori_context[0, :, :, j] - f_steg_context[0, :, :, j]
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    plt.imshow(diff)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
plt.savefig("feature difference.png", dpi=5000)

plt.figure()
for j in range(81):
    print(j)
    plt.subplot(9, 9, j + 1)
    sns.distplot(f_ori_context[:10, :, :, j].reshape(-1), kde=True, bins=30)
    sns.distplot(f_steg_context[:10, :, :, j].reshape(-1), kde=True, bins=30)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
plt.savefig("feature_hist.png", dpi=500)

train_vector = np.concatenate((f_ori_context[diff_map != 0], f_steg_context[diff_map != 0]), axis=0)
train_label = np.concatenate((0 * np.ones(np.sum(diff_map != 0)), 1 * np.ones(np.sum(diff_map != 0))))
idx = np.random.permutation(len(train_label))
train_vector = train_vector[idx]
train_label = train_label[idx]

mi = mutual_info_regression(train_vector[:50000], train_label[:50000])

channels = [3, 7, 14, 20, 22, 23, 25, 26, 27, 33, 40, 46, 47, 48, 51, 52,
            58, 68, 71, 76, 77, 80]
