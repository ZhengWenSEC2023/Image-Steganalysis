import gc
import pickle
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from scipy import stats

#########################################################
#   STEP 2: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT   #
#     EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT     #
#########################################################
for k in range(1, 9):
    # (11249863 * 2, 1, 1, 22)
    print(k)
    cur_features = np.load("/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(k) + "_PIXEL_1.npy")
    cur_features = np.squeeze(cur_features) # (11249863 * 2, 22-k-)  (steg + cover)
    train_features = np.concatenate((cur_features[:500000], cur_features[11249863: 11249863 + 500000]), )
    test_features = np.concatenate((cur_features[600000: 600000 + 100000], cur_features[11249863 + 600000: 11249863 + 600000 + 100000]), )
    del cur_features
    gc.collect()
    train_labels = np.concatenate((1 * np.ones(500000), 0 * np.ones(500000)), axis=0)
    test_labels = np.concatenate((1 * np.ones(100000), 0 * np.ones(100000)), axis=0)
    idx = np.random.permutation(len(train_labels))
    train_features = train_features[idx]
    train_labels = train_labels[idx]

    param = {
        "n_estimators": [1000],
        'learning_rate': stats.uniform(0.01, 0.19),
        'subsample': stats.uniform(0.3, 0.6),
        'max_depth': [4, 5, 6, 7, 8],
        'colsample_bytree': stats.uniform(0.5, 0.4),
        'min_child_weight': [1, 2, 3, 4, 5]
    }

    folds = 4
    param_comb = 3
    np.random.seed(23)

    xgb = XGBClassifier()
    # K-Fold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    random_search = RandomizedSearchCV(xgb, param_distributions=param, n_iter=param_comb, scoring='roc_auc',
                                       n_jobs=8, cv=skf.split(train_features, train_labels), random_state=1001)
    random_search.fit(train_features, train_labels)
    cur_best = random_search.best_estimator_

    f = open(str(k) + "_feature_" + "_XGBOOST_PIXEL.pkl", "wb")
    pickle.dump(cur_best, f)
    f.close()

    print("TRAIN SCORE:", cur_best.score(train_features, train_labels))
    print("TEST SCORE:", cur_best.score(test_features, test_labels))
