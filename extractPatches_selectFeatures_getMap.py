import os
import pickle
import numpy as np
import cv2
from pixelhop2 import Pixelhop2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
from sklearn.feature_selection import mutual_info_regression

#######################################################################
# STEP 1: TRAIN PIXELHOP BY STEGO IMAGE ONLY CONTAINING STEGO PATCHES #
#    TRAINED PIXELHOP UNIT IS SAVED (A SINGLE PIXELHOP IS TRAINED)    #
#######################################################################

P_TRAIN_NUM_TOTAL = 2000

print(P_TRAIN_NUM_TOTAL)
np.random.seed(23)
ori_train_img = []
steg_train_img = []
ori_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'
file_names = os.listdir(ori_img_path)
file_names.sort(key=lambda x: int(x[:-4]))

steg_patches = []
cover_patches = []
count = 0

for file_name in file_names:
    print(count)
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)[:, :, None]
    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)[:, :, None]
    diff_map = np.squeeze(ori_img - steg_img)
    for i in range(2, 254):
        for j in range(2, 254):
            if diff_map[i, j] != 0:
                steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
                cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
    count += 1
    if count == P_TRAIN_NUM_TOTAL:
        break

steg_patches = np.array(steg_patches)
cover_patches = np.array(cover_patches)

# PixlHop ++
def Shrink(X, shrinkArg, max_pooling=True, padding=True):
    X = view_as_windows(X, (1, 5, 5, 1))
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
    return X

# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X


# set args for PixelHop++
SaabArgs = [{'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None}]
shrinkArgs = [{'func': Shrink, 'win': 5}]
concatArg = {'func': Concat}

p2 = Pixelhop2(depth=1, TH1=1e-20, TH2=1e-30, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs,
               concatArg=concatArg).fit(steg_patches)

f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopStegoPts_5_5_1st_layer.pkl", 'wb')
# f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopStegoPts_7_7_1st_layer.pkl", 'wb')
pickle.dump(p2, f)
f.close()

# features_cover = p2.transform(cover_patches)
# features_steg = p2.transform(steg_patches)
#
# features_cover = np.squeeze(features_cover)
# features_steg = np.squeeze(features_steg)
#
# idx = np.random.permutation(len(features_cover))
# features_cover = features_cover[idx]
# features_steg = features_steg[idx]
#
# NUM_SELECT = 100000
#
# train_features = np.concatenate((features_cover[:NUM_SELECT], features_steg[:NUM_SELECT]), axis=0)
# train_labels = np.concatenate((0 * np.ones(len(features_cover[:NUM_SELECT])), 1 * np.ones(len(features_steg[:NUM_SELECT]))), axis=0)
# idx = np.random.permutation(len(train_features))
# train_features = train_features[idx]
# train_labels = train_labels[idx]
#
del cover_patches, steg_patches
# del features_cover, features_steg
#
# # mi = mutual_info_regression(train_features, train_labels)
mi = np.array([0.00E+00, 0.00E+00, 0.00E+00, 7.00E-04, 5.42E-04,
               0.00E+00, 1.54E-05, 0.00E+00, 2.81E-03, 1.27E-03,
               2.15E-04, 0.00E+00, 4.79E-04, 0.00E+00, 0.00E+00,
               0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00,
               8.04E-04, 0.00E+00, 0.00E+00, 1.15E-03, 0.00E+00])

selected = np.where(mi != 0)[0]


steg_patches = []
cover_patches = []
count = 0

for file_name in file_names:
    print(count)
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)[:, :, None]
    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)[:, :, None]
    diff_map = np.squeeze(ori_img - steg_img)
    for i in range(2, 62):
        for j in range(2, 62):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(66, 126):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(130, 190):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(194, 254):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])

    for i in range(66, 126):
        for j in range(2, 62):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(66, 126):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(130, 190):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(194, 254):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])

    for i in range(130, 190):
        for j in range(2, 62):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(66, 126):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(130, 190):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(194, 254):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])

    for i in range(194, 254):
        for j in range(2, 62):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(66, 126):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(130, 190):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
        for j in range(194, 254):
            steg_patches.append(steg_img[i - 2: i + 3, j - 2: j + 3])
            cover_patches.append(ori_img[i - 2: i + 3, j - 2: j + 3])
    count += 1
    if count == P_TRAIN_NUM_TOTAL:
        break

steg_patches = np.array(steg_patches)
cover_patches = np.array(cover_patches)

features_cover = p2.transform(cover_patches)
features_steg = p2.transform(steg_patches)

# shape should be (2000, 16, x, x, 25)

features_cover = np.squeeze(features_cover[0])
features_steg = np.squeeze(features_steg[0])

features_cover = features_cover.reshape((2000, 16, 60, 60, 25))
features_steg = features_steg.reshape((2000, 16, 60, 60, 25))

features_cover = features_cover.reshape((2000 * 16, 60, 60, 25))
features_steg = features_steg.reshape((2000 * 16, 60, 60, 25))

np.save("week_12_feature_cover_px1,npy", features_cover)
np.save("week_12_feature_steg_px1,npy", features_steg)

selected_cover = features_cover[:, :, :, selected]
selected_steg = features_steg[:, :, :, selected]

np.save("week_12_feature_cover_selected_px1,npy", selected_cover)
np.save("week_12_feature_steg_selected_px1,npy", selected_steg)
