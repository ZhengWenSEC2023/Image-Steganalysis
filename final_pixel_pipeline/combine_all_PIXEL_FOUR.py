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
total_train_features = []
for k in range(22):
    # (11249863 * 2, 1, 1, 22)
    print(k)
    cur_features = np.load(
        "/mnt/zhengwen/image_steganalysis/dataset/codes/end_features_from_" + str(k) + "_PIXEL_1_sample_ALL.npy")
    cur_features = np.squeeze(cur_features)  # (5406302 * 2, 22-k-)  (steg + cover)
    train_features = np.concatenate((cur_features[:500000], cur_features[5406302: 5406302 + 500000]), )
    total_train_features.append(train_features)
    del cur_features
    gc.collect()

total_train_features = np.concatenate(total_train_features, axis=1)
train_labels = np.concatenate((1 * np.ones(500000), 0 * np.ones(500000)), axis=0)
idx = np.random.permutation(len(total_train_features))
total_train_features = total_train_features[idx]
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
                                   n_jobs=12, cv=skf.split(total_train_features, train_labels), random_state=1001)
random_search.fit(total_train_features, train_labels)
cur_best = random_search.best_estimator_

f = open("total_feature_" + "_XGBOOST_PIXEL_sample.pkl", "wb")
pickle.dump(cur_best, f)
f.close()

print("TRAIN SCORE:", cur_best.score(total_train_features, train_labels))
