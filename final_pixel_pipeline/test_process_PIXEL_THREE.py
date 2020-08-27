import gc
import pickle
import numpy as np
from xgboost import XGBClassifier

#########################################################
#   STEP 2: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT   #
#     EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT     #
#########################################################
for k in range(1):
    # (11249863 * 2, 1, 1, 22)
    print(k)
    cur_features = np.load("/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(k) + "_PIXEL_1_TEST.npy")  # ((2463760, 1, 1, 18))
    cur_features = np.squeeze(cur_features) # (1231880 * 2, 22-k-)  (steg + cover)
    test_features = np.concatenate((cur_features[:100000], cur_features[1231880: 1231880 + 100000]), )
    del cur_features
    gc.collect()
    test_labels = np.concatenate((1 * np.ones(100000), 0 * np.ones(100000)), axis=0)
    xgb = XGBClassifier()
    # K-Fold

    f = open(str(k) + "_feature_" + "_XGBOOST_PIXEL.pkl", "rb")
    cur_best = pickle.load(f)
    f.close()

    print("TEST SCORE:", cur_best.score(test_features, test_labels))
