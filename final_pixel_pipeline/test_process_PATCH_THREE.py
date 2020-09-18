import pickle
import numpy as np

#########################################################
#   STEP 2: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT   #
#     EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT     #
#########################################################
for k in range(9):
    cur_features = np.load("/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(k) + "_test.npy")
    cur_features = cur_features.reshape((3200, -1, cur_features.shape[-1]))  # 3200, 144, 56
    cur_features = np.moveaxis(cur_features, -1, 0)       # 56, 3200, 144
    for i in range(cur_features.shape[0]):
        print(k, i)
        cur_sub_features = cur_features[i, :, :]
        cur_sub_labels = np.concatenate( (0 * np.ones(1600), 1 * np.ones(1600)), axis=0 )
        f = open(str(k) + "_feature_" + str(i) + "_th" + "_XGBOOST.pkl", "rb")
        cur_best = pickle.load(f)
        f.close()
        print(cur_best.score(cur_sub_features, cur_sub_labels))
