from pixelhop2 import Pixelhop2
import gc
import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from scipy import stats

#########################################################
#   STEP 2: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT   #
#     EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT     #
#########################################################
for k in range(1):
    cur_features = np.load("/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(k) + ".npy")
    cur_features = cur_features.reshape((64000, -1, cur_features.shape[-1]))  # 64000, 144, 56
    cur_features = np.moveaxis(cur_features, -1, 0)       # 56, 64000, 144
    for i in range(cur_features.shape[0]):
        print(k, i)
        cur_sub_features = cur_features[i, :, :]
        cur_sub_labels = np.concatenate( (0 * np.ones(32000), 1 * np.ones(32000)), axis=0 )
        idx = np.random.permutation(len(cur_sub_features))
        cur_sub_features = cur_sub_features[idx]
        cur_sub_labels = cur_sub_labels[idx]

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
                                           n_jobs=8, cv=skf.split(cur_sub_features, cur_sub_labels), random_state=1001)
        random_search.fit(cur_sub_features, cur_sub_labels)
        cur_best = random_search.best_estimator_

        f = open(str(k) + "_feature_" + str(i) + "_th" + "_XGBOOST.pkl", "wb")
        pickle.dump(cur_best, f)
        f.close()

