import gc
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#########################################################
#   STEP 2: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT   #
#     EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT     #
#########################################################
total_mi = []
NUM_SELECT = 500000
total_feature = []
for k in range(22):
    # (11249863 * 2, 1, 1, 22)
    gc.collect()
    print(k)
    cur_features = np.load(
        "/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(k) + "_PIXEL_1_sample_ALL.npy")
    cur_features = np.squeeze(cur_features)  # (5406302 * 2, 22-k-)  (steg + cover)
    cover_feature = cur_features[:5406302, :]
    steg_feature = cur_features[5406302:, :]

    idx = np.random.permutation(len(cover_feature))
    cover_feature = cover_feature[idx]
    steg_feature = steg_feature[idx]

    train_features = np.concatenate((steg_feature[:NUM_SELECT], cover_feature[:NUM_SELECT]), axis=0)

    idx = np.random.permutation(len(train_features))
    train_features = train_features[idx]
    total_feature.append(train_features)

    #
total_feature = np.concatenate(total_feature, axis=1)
train_labels = np.concatenate(
        (1 * np.ones(NUM_SELECT), 0 * np.ones(NUM_SELECT)), axis=0)

rf = RandomForestClassifier(n_estimators=150, max_depth=13, min_samples_split=110, min_samples_leaf=20,
                            max_features=5)
rf.fit(total_feature, train_labels)
    # xgb = XGBClassifier(subsample=0.7, colsample_bytree=0.7, eta=0.1, n_estimators=550,
    #                     min_child_weight=5, max_depth=4, gamma=0.1)
    # xgb.fit(train_features, train_labels)
    # # 3:38
    # break
    # (array([ 4,  5,  9, 10, 13, 20, 22, 39, 41, 45, 46, 47, 49, 56, 62, 63, 64,
    #        65, 66, 70, 74, 79, 80]),)
    # (array([ 2,  4,  5, 13, 14, 15, 30, 39, 41, 43, 51, 52, 61, 62, 67, 70, 72,
    #         73, 76, 78]),)
    # (array([ 2,  5,  6,  9, 14, 22, 24, 25, 29, 32, 34, 36, 37, 38, 39, 40, 43,
    #         48, 50, 51, 60, 62, 66, 69, 71, 72, 73, 76, 77, 80]),)
    # (array([ 0,  1,  4,  5,  6,  9, 10, 15, 20, 22, 23, 32, 39, 42, 47, 51, 52,
    #         54, 55, 56, 61, 62, 63, 64, 65, 67, 68, 71, 74, 77, 79, 80]),)