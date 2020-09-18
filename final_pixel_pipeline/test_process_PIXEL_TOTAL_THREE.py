import gc
import pickle
import numpy as np
from xgboost import XGBClassifier

#########################################################
#   STEP 2: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT   #
#     EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT     #
#########################################################
# f = open("image_wise_final.pkl", "rb")
# features_final_image_wise = pickle.load(f)
# f.close()

f = open("image_wise_final_rho.pkl", "rb")
features_final_image_wise = pickle.load(f)
f.close()

precentage_stego_stego = []
precentage_stego_cover = []

for i in range(200):
    print(i)
    total_features = []
    for k in range(22):
        # (11249863 * 2, 1, 1, 22)
        # cur_features = np.load("/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(k) + "_PIXEL_1_TEST_sample.npy")  # ((2463760, 1, 1, 18))
        cur_features = features_final_image_wise[i][k]
        cur_features = np.squeeze(cur_features) # (1231880 * 2, 22-k-)  (steg + cover)
        # test_features = np.concatenate((cur_features[:100000], cur_features[1231880: 1231880 + 100000]), )
        test_features = cur_features
        # del cur_features

        gc.collect()
        total_features.append(test_features)

    total_features = np.concatenate(total_features, axis=1)
    test_labels = np.concatenate((1 * np.ones(total_features.shape[0] // 2), 0 * np.ones(total_features.shape[0] // 2)), axis=0)
    xgb = XGBClassifier()
    # K-Fold

    f = open("total_feature_" + "_XGBOOST_PIXEL_sample.pkl", "rb")
    cur_best = pickle.load(f)
    f.close()

    # print("TEST SCORE:", cur_best.score(total_features, test_labels))
    res = cur_best.predict(total_features)
    precentage_stego_stego.append(np.sum(res[:res.shape[0] // 2] == 1) / (res.shape[0] // 2))
    precentage_stego_cover.append(np.sum(res[res.shape[0] // 2:] == 1) / (res.shape[0] // 2))

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.distplot(precentage_stego_cover, label="cover")
sns.distplot(precentage_stego_stego, label="stego")
plt.legend()
