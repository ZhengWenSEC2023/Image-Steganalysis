import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

#################################################################
#   STEP 4: USE SVM TO ENSEMBLE THE RESULT OF PROBABILITY VEC   #
# A TRAINED SVM IS SAVED AND TRAIN SCORE && TEST SCORE IS GIVEN #
#################################################################
CHANNEL_RANGE = [0, 15, 15 + 22, 15 + 22 + 28, 15 + 22 + 28 + 39]


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

f = open("ensemble_clf_PH4.pkl", "rb")
clf_list = pickle.load(f)
f.close()

f = open("PixelHopUniform_4PH.pkl", 'rb')
p2 = pickle.load(f)
f.close()

folds = 5
param_comb = 20
np.random.seed(23)


train_f_ori_vectors = np.load("week8_train_ori_feature_vec_PH4.npy")
train_f_steg_vectors = np.load("week8_train_steg_feature_vec_PH4.npy")
test_f_ori_vectors = np.load("week8_test_ori_feature_vec_PH4.npy")
test_f_steg_vectors = np.load("week8_test_steg_feature_vec_PH4.npy")

train_label = np.concatenate((0 * np.ones(len(train_f_ori_vectors)), 1 * np.ones(len(train_f_steg_vectors))), axis=0)
test_label = np.concatenate((0 * np.ones(len(test_f_ori_vectors)), 1 * np.ones(len(test_f_steg_vectors))), axis=0)

train_pred_prob = []
test_pred_prob = []
for i in range(1, len(CHANNEL_RANGE)):
    print(i)
    train_ori_vec = train_f_ori_vectors[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    train_steg_vec = train_f_steg_vectors[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    train_sample = np.concatenate((train_ori_vec, train_steg_vec), axis=0)

    test_ori_vec = test_f_ori_vectors[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    test_steg_vec = test_f_steg_vectors[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    test_sample = np.concatenate((test_ori_vec, test_steg_vec), axis=0)

    clf = clf_list[i - 1]

    cur_train_prob = clf.predict_proba(train_sample)[:, 1]
    cur_test_prob = clf.predict_proba(test_sample)[:, 1]
    train_pred_prob.append(cur_train_prob)
    test_pred_prob.append(cur_test_prob)

# Final Classifier

train_pred_prob = np.array(train_pred_prob)
test_pred_prob = np.array(test_pred_prob)

train_pred_prob = train_pred_prob.transpose()
test_pred_prob = test_pred_prob.transpose()

idx = np.random.permutation(len(train_pred_prob))
train_pred_prob = train_pred_prob[idx]
train_label = train_label[idx]

params = {
    'min_child_weight': [1, 2, 4, 5, 6, 7, 8, 10, 11, 14, 15, 17, 21],
    'gamma':            [0.5, 1, 1.5, 2, 5],
    'subsample':        [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate':    [0.1, 0.01, 0.2]
}

folds = 5
param_comb = 10
np.random.seed(23)

xgb = XGBClassifier()
# K-Fold
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
                                   n_jobs=-1, cv=skf.split(train_pred_prob, train_label), random_state=1001)
random_search.fit(train_pred_prob, train_label)
best_final = random_search.best_estimator_

f = open("final_XG_BOOST", "wb")
pickle.dump(best_final, f)
f.close()

print("TRAIN SCORE:", best_final.score(train_pred_prob, train_label))
print("TEST SCORE:", best_final.score(test_pred_prob, test_label))
