import os
import pickle
import numpy as np
import cv2
from final_pixel_pipeline.pixelhop2 import Pixelhop2
from skimage.measure import block_reduce
from skimage.util import view_as_windows

#########################################################
# STEP 1: TRAIN PIXELHOP BY STEGO IMAGE AND COVER IMAGE #
#            TRAINED PIXELHOP UNIT IS SAVED             #
#########################################################


P_TRAIN_NUM_TOTAL = 1000
print(P_TRAIN_NUM_TOTAL)
np.random.seed(23)
ori_train_img = []
steg_train_img = []
ori_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'
file_names = os.listdir(ori_img_path)
file_names.sort(key=lambda x: int(x[:-4]))

count = 0
for file_name in file_names:
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)[:, :, None]
    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)[:, :, None]
    ori_train_img.append(ori_img)
    steg_train_img.append(steg_img)
    count += 1
    if count == P_TRAIN_NUM_TOTAL:
        break

ori_train_img = np.array(ori_train_img)
steg_train_img = np.array(steg_train_img)
diff_map = ori_train_img - steg_train_img
train_img = np.concatenate((ori_train_img, steg_train_img), axis=0)
train_label = np.concatenate((0 * np.ones(len(ori_train_img)), 1 * np.ones(len(steg_train_img))), axis=0)
idx = np.random.permutation(train_img.shape[0])
train_img = train_img[idx]
train_label = train_label[idx]


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


# set args for PixelHop++
SaabArgs = [{'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None},
            {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None},
            {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None},
            {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None}]
shrinkArgs = [{'func': Shrink, 'win': 9},
              {'func': Shrink, 'win': 5},
              {'func': Shrink, 'win': 5},
              {'func': Shrink, 'win': 5}]
concatArg = {'func': Concat}

# PixlHop ++
p2 = Pixelhop2(depth=1, TH1=0.005, TH2=0.0005, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs,
               concatArg=concatArg).fit(train_img)

# p2.construct_count()

# f = open("PixelHopUniform.pkl", 'wb')  # 3 PixelHop, win: 5, TH1:0.005, TH2:0.005, CH1: 15, CH2: 20, CH3: 25, NUM_TRAIN=500
f = open("PixelHopUniform_4PH_HOP1_RETRAINED.pkl", 'wb')
pickle.dump(p2, f)
f.close()

