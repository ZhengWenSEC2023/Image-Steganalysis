import pickle
import numpy as np
import cv2
from pixelhop2 import Pixelhop2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import gc

#########################################################
#   STEP 2: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT   #
#     EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT     #
#########################################################

def Shrink(X, shrinkArg, max_pooling=True, padding=False):

    win = shrinkArg['win']
    max_pooling = shrinkArg['max_pooling']
    padding = shrinkArg['padding']
    if max_pooling:
        X = block_reduce(X, (1, 2, 2, 1), np.average)
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
            {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None}]
shrinkArgs = [{'func': Shrink, 'win': 5, 'max_pooling': False, 'padding': False},
              {'func': Shrink, 'win': 5, 'max_pooling': True, 'padding': False}]
concatArg = {'func': Concat}

selected_cover = np.load("week_12_feature_cover_selected_px1,npy.npy")   # (32000, 60, 60, 9)
selected_steg = np.load("week_12_feature_steg_selected_px1,npy.npy")     # (32000, 60, 60, 9)

train_selected = np.concatenate((selected_steg, selected_cover), axis=0)

# PixlHop ++
for i in range(9):  # pixelHop 0, 1: {TH1: 0.05, TH2: 0.005} finished   # features: 0 finished
    print(i)
    p2_1 = Pixelhop2(depth=2, TH1=0.05, TH2=0.01, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs,
                   concatArg=concatArg).fit(train_selected[:, :, :, i][:, :, :, None])
    f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopStegoPts_5_5_2nd_3rd_layer_" + str(i) + ".pkl", 'wb')
    pickle.dump(p2_1, f)
    f.close()

    feature_p2_1 = p2_1.transform(train_selected[:, :, :, i][:, :, :, None])
    cur_features = block_reduce(feature_p2_1[1], (1, 2, 2, 1), np.average)
    print(i, cur_features.shape)
    np.save("/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(i) + ".npy", cur_features)
    del p2_1, feature_p2_1, cur_features
    gc.collect()

# for i in range(9):
#     print(i)
#     f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopStegoPts_5_5_2nd_layer_" + str(i) + ".pkl", 'rb')
#     p2_1 = pickle.load(f)
#     f.close()
#     feature_p2_1 = p2_1.transform(train_selected[:, :, :, i][:, :, :, None])[0]
