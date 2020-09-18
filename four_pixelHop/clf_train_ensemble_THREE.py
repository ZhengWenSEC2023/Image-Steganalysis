import pickle
import numpy as np
import cv2
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from skimage.measure import block_reduce
from skimage.util import view_as_windows
from sklearn.svm import SVC

#########################################################
#    STEP 3: ENSEMBLE CLASSIFIER BEHIND EACH PIXELHOP   #
#      A LIST OF TRAINED XGBOOST CLASSIFIER IS SAVED    #
#########################################################

CHANNEL_RANGE = [0, 25 * 9, 25 * 9 + 22, 25 * 9 + 22 + 28, 25 * 9 + 22 + 28 + 39]


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

def context_resize(context):
    context = np.moveaxis(context, -1, 1)
    new_context = []
    for i in range(context.shape[0]):
        for j in range(context.shape[1]):
            new_context.append(cv2.resize(context[i, j], (256, 256), interpolation=cv2.INTER_NEAREST))
    new_context = np.array(new_context)
    new_context = np.reshape(new_context, (context.shape[0], context.shape[1], 256, 256))
    new_context = np.moveaxis(new_context, 1, -1)
    return new_context

f = open("PixelHopUniform_4PH.pkl", 'rb')
p2 = pickle.load(f)
f.close()

train_f_ori_vectors = np.load("week8_train_ori_feature_vec_PH4.npy")
train_f_steg_vectors = np.load("week8_train_steg_feature_vec_PH4.npy")

params = {
    'min_child_weight': [1, 2, 4, 5, 6, 7, 8, 10, 11, 14, 15, 17, 21],
    'gamma':            [0.5, 1, 1.5, 2, 5],
    'subsample':        [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate':    [0.1, 0.01, 0.2]
}

folds = 5
param_comb = 10
clf_list = []
np.random.seed(23)

for i in range(1, len(CHANNEL_RANGE)):
    print(i)
    train_ori_vec = train_f_ori_vectors[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    train_steg_vec = train_f_steg_vectors[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    train_sample = np.concatenate((train_ori_vec, train_steg_vec), axis=0)
    train_label = np.concatenate((0 * np.ones(len(train_ori_vec)), 1 * np.ones(len(train_steg_vec))), axis=0)
    idx = np.random.permutation(len(train_sample))
    train_sample = train_sample[idx]
    train_label = train_label[idx]
    # XGB
    xgb = XGBClassifier()
    # K-Fold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
                                       n_jobs=-1, cv=skf.split(train_sample, train_label), random_state=1001)
    random_search.fit(train_sample, train_label)
    clf_list.append(random_search.best_estimator_)
    print("end", i)

# f = open("ensemble_clf.pkl", "wb")  # for 3 pixelHOP with 1000 training samples
f = open("ensemble_clf_PH4.pkl", "wb")
pickle.dump(clf_list, f)
f.close()

# FINAL CLASSIFIER (TILL 7/14 UNIMPLEMENTED BELOW)

# train_pred_prob = []
# for i in range(len(counts)):
#     train_ori_vec = train_f_ori_vectors[:, counts[i][0]: counts[i][-1]]
#     train_steg_vec = train_f_steg_vectors[:, counts[i][0]: counts[i][-1]]
#     train_sample = np.concatenate((train_ori_vec, train_steg_vec), axis=0)
#     clf = clf_list[i]
#     train_pred_prob.append(clf.predict_proba(train_sample)[:, 1])
#
# train_pred_prob = np.array(train_pred_prob).transpose()
# train_label = np.concatenate((0 * np.ones(len(train_f_ori_vectors)), 1 * np.ones(len(train_f_steg_vectors))), axis=0)
#
# idx = np.random.permutation(len(train_pred_prob))
# train_pred_prob = train_pred_prob[idx]
# train_label = train_label[idx]

# svm_clf = SVC()
# parameters = {
#     'C':     np.logspace(-3, 1, 6),
#     'gamma': np.logspace(-3, 1, 6)
# }
#
# skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
# random_search = RandomizedSearchCV(svm_clf, param_distributions=parameters, n_iter=param_comb, scoring='roc_auc',
#                                    n_jobs=-1, cv=skf.split(train_pred_prob, train_label), random_state=1001)
# random_search.fit(train_pred_prob, train_label)
#
# svm_best = random_search.best_estimator_
#
# print(svm_best.score(train_pred_prob, train_label))
#
