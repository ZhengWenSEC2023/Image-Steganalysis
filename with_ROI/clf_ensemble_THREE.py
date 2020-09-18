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

#################################################################
#   STEP 3: USE SVM TO ENSEMBLE THE RESULT OF PROBABILITY VEC   #
# A TRAINED SVM IS SAVED AND TRAIN SCORE && TEST SCORE IS GIVEN #
#################################################################

CHANNEL_RANGE = [i * 80 for i in range(10)]

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

f = open("ensemble_clf_singPH1.pkl", "rb")
clf_list = pickle.load(f)
f.close()

f = open("PixelHopUniform_singPH.pkl", 'rb')
p2 = pickle.load(f)
f.close()

train_f_ori_vectors = np.load("week8_train_ori_context_1_PH4_RETRAINED.npy")
train_f_steg_vectors = np.load("week8_train_steg_context_1_PH4_RETRAINED.npy")

train_label = np.concatenate((0 * np.ones(len(train_f_ori_vectors)), 1 * np.ones(len(train_f_steg_vectors))), axis=0)

train_pred_prob = []
for i in range(1, len(CHANNEL_RANGE)):
    print(i)
    train_ori_vec = train_f_ori_vectors[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    train_steg_vec = train_f_steg_vectors[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    train_sample = np.concatenate((train_ori_vec, train_steg_vec), axis=0)

    clf = clf_list[i - 1]

    cur_train_prob = clf.predict_proba(train_sample)[:, 1]
    train_pred_prob.append(cur_train_prob)

# Final Classifier

train_pred_prob = np.array(train_pred_prob)

train_pred_prob = train_pred_prob.transpose()

idx = np.random.permutation(len(train_pred_prob))
train_pred_prob = train_pred_prob[idx]
train_label = train_label[idx]

param = {
    "n_estimators": [1000],
    'learning_rate': stats.uniform(0.01, 0.19),
    'subsample': stats.uniform(0.3, 0.6),
    'max_depth': [4, 5, 6, 7, 8],
    'colsample_bytree': stats.uniform(0.5, 0.4),
    'min_child_weight': [1, 2, 3, 4, 5]
    }

folds = 5
param_comb = 5
np.random.seed(23)

xgb = XGBClassifier()
# K-Fold
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
random_search = RandomizedSearchCV(xgb, param_distributions=param, n_iter=param_comb, scoring='roc_auc',
                                   n_jobs=-1, cv=skf.split(train_pred_prob, train_label), random_state=1001)
random_search.fit(train_pred_prob, train_label)
best_final = random_search.best_estimator_

f = open("final_XG_BOOST_SINGLE_PXH.pkl", "wb")
pickle.dump(best_final, f)
f.close()

print("TRAIN SCORE:", best_final.score(train_pred_prob, train_label))
