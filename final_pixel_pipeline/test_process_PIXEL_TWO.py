import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows


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
shrinkArgs = [{'func': Shrink, 'win': 3, 'max_pooling': False, 'padding': False},
              {'func': Shrink, 'win': 3, 'max_pooling': False, 'padding': False}]
concatArg = {'func': Concat}

# selected_cover = np.load("week_12_feature_cover_selected_px1_PIXEL_TEST.npy")   # (11249863, 5, 5, 9)
# selected_steg = np.load("week_12_feature_steg_selected_px1_PIXEL_TEST.npy")     # (11249863, 5, 5, 9)

# selected_cover = np.load("week_12_feature_cover_selected_px1_PIXEL_TEST_sample.npy")   # (11249863, 5, 5, 9)
# selected_steg = np.load("week_12_feature_steg_selected_px1_PIXEL_TEST_sample.npy")     # (11249863, 5, 5, 9)
#
# test_selected = np.concatenate((selected_steg, selected_cover), axis=0)

# f = open("image_wise_cover.pkl", "rb")
# features_cover = pickle.load(f)
# f.close()
#
# f = open("image_wise_steg.pkl", "rb")
# features_steg = pickle.load(f)
# f.close()

f = open("image_wise_cover_rho.pkl", "rb")
features_cover = pickle.load(f)
f.close()

f = open("image_wise_steg_rho.pkl", "rb")
features_steg = pickle.load(f)
f.close()

features_final_image_wise = []
for k in range(200):
    print(k)
    cur_cover = features_cover[k]
    cur_stego = features_steg[k]
    test_selected = np.concatenate((cur_stego, cur_cover), axis=0)
    # PixlHop ++
    features_final = []
    for i in range(22):  # pixelHop 0, 1: {TH1: 0.05, TH2: 0.005} finished   # features: 0 finished

        f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopStegoPts_5_5_2nd_3rd_layer_" + str(i) + "_PIXEL_sample.pkl", 'rb')
        p2_1 = pickle.load(f)
        f.close()
        feature_p2_1 = p2_1.transform(test_selected[:, :, :, i][:, :, :, None])
        print(i, feature_p2_1[0].shape)
        print(i, feature_p2_1[1].shape)
        print("/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(i) + "_PIXEL_0_TEST_sample.npy")
        # np.save("/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(i) + "_PIXEL_0_TEST_sample.npy", feature_p2_1[0])
        # np.save("/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(i) + "_PIXEL_1_TEST_sample.npy", feature_p2_1[1])
        # del p2_1, feature_p2_1
        # gc.collect()
        features_final.append(feature_p2_1[1])
    features_final_image_wise.append(features_final)

# f = open("image_wise_final.pkl", "wb")
# pickle.dump(features_final_image_wise, f)
# f.close()

f = open("image_wise_final_rho.pkl", "wb")
pickle.dump(features_final_image_wise, f)
f.close()

# for i in range(9):
#     print(i)
#     f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopStegoPts_5_5_2nd_layer_" + str(i) + ".pkl", 'rb')
#     p2_1 = pickle.load(f)
#     f.close()
#     feature_p2_1 = p2_1.transform(train_selected[:, :, :, i][:, :, :, None])[0]
