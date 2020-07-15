import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

#################################################################
#   STEP 4: USE SVM TO ENSEMBLE THE RESULT OF PROBABILITY VEC   #
# A TRAINED SVM IS SAVED AND TRAIN SCORE && TEST SCORE IS GIVEN #
#################################################################


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

f = open("ensemble_clf.pkl", "rb")
clf_list = pickle.load(f)
f.close()

f = open("PixelHopUniform.pkl", 'rb')
p2 = pickle.load(f)
f.close()

counts = p2.counts
folds = 5
param_comb = 20
np.random.seed(23)

for i in range(1, len(counts)):
    counts[i] += counts[i - 1][-1]

train_f_ori_vectors = np.load("week8_train_ori_feature_vec_PH4.npy")
train_f_steg_vectors = np.load("week8_train_steg_feature_vec_PH4.npy")
test_f_ori_vectors = np.load("week8_test_ori_feature_vec.npy")
test_f_steg_vectors = np.load("week8_test_steg_feature_vec.npy")

train_label = np.concatenate((0 * np.ones(len(train_f_ori_vectors)), 1 * np.ones(len(train_f_steg_vectors))), axis=0)
test_label = np.concatenate((0 * np.ones(len(test_f_ori_vectors)), 1 * np.ones(len(test_f_steg_vectors))), axis=0)

train_pred_prob = []
test_pred_prob = []
for i in range(len(counts)):
    print(i)
    train_ori_vec = train_f_ori_vectors[:, counts[i][0]: counts[i][-1]]
    train_steg_vec = train_f_steg_vectors[:, counts[i][0]: counts[i][-1]]
    train_sample = np.concatenate((train_ori_vec, train_steg_vec), axis=0)

    test_ori_vec = test_f_ori_vectors[:, counts[i][0]: counts[i][-1]]
    test_steg_vec = test_f_steg_vectors[:, counts[i][0]: counts[i][-1]]
    test_sample = np.concatenate((test_ori_vec, test_steg_vec), axis=0)

    clf = clf_list[i]

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

svm_clf = SVC()
parameters = {
    'C':     np.logspace(-3, 1, 6),
    'gamma': np.logspace(-3, 1, 6)
}

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
random_search = RandomizedSearchCV(svm_clf, param_distributions=parameters, n_iter=param_comb, scoring='roc_auc',
                                   n_jobs=-1, cv=skf.split(train_pred_prob, train_label), random_state=1001)
random_search.fit(train_pred_prob, train_label)
svm_best = random_search.best_estimator_

f = open("final_SVM_PH4.pkl", "wb")
pickle.dump(svm_best, f)
f.close()

print("TRAIN SCORE:", svm_best.score(train_pred_prob, train_label))
print("TEST SCORE:", svm_best.score(test_pred_prob, test_label))
