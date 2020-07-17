import pickle
import numpy as np
import cv2
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from skimage.measure import block_reduce
from skimage.util import view_as_windows
from sklearn.svm import SVC
########################################################
# -------------------ADDITIONAL----------------------- #
# TEST WHETHER THE UNCHANGED POINT IS HALF HALF OR NOT #
########################################################

RAD = 1
RAD_OUT = 2
NUM_UNIFORM = 20000000
NUM_HARD = 20000000
CHANNEL_RANGE = [0, 15, 15 + 22, 15 + 22 + 28, 15 + 22 + 28 + 39]

f_ori_context_1 = np.load("week8_test_ori_context_1_PH4.npy")
f_ori_context_2 = np.load("week8_test_ori_context_2_PH4.npy")
f_ori_context_3 = np.load("week8_test_ori_context_3_PH4.npy")
f_ori_context_4 = np.load("week8_test_ori_context_4_PH4.npy")
f_steg_context_1 = np.load("week8_test_steg_context_1_PH4.npy")
f_steg_context_2 = np.load("week8_test_steg_context_2_PH4.npy")
f_steg_context_3 = np.load("week8_test_steg_context_3_PH4.npy")
f_steg_context_4 = np.load("week8_test_steg_context_4_PH4.npy")
diff_map = np.load("week8_diff_map_test_PH4.npy")
diff_map = abs(diff_map)

test_ori_context = np.concatenate((f_ori_context_1, f_ori_context_2, f_ori_context_3, f_ori_context_4), axis=-1)
test_steg_context = np.concatenate((f_steg_context_1, f_steg_context_2, f_steg_context_3, f_steg_context_4), axis=-1)

# UNIFORMLY
# train_unchanged_ori_context = train_ori_context.copy()[diff_map == 0]
# train_unchanged_steg_context = train_steg_context.copy()[diff_map == 0]
#
# idx = np.random.permutation(len(train_unchanged_ori_context))
# train_unchanged_ori_context = train_unchanged_ori_context[idx][:NUM_UNIFORM]
# idx = np.random.permutation(len(train_unchanged_steg_context))
# train_unchanged_steg_context = train_unchanged_steg_context[idx][:NUM_UNIFORM]

for k in range(diff_map.shape[0]):
    print(k)
    visit_map = np.zeros(diff_map[k].shape)
    for i in range(diff_map[k].shape[0]):
        for j in range(diff_map[k].shape[1]):
            if diff_map[k, i, j] != 0 and visit_map[i, j] == 0:
                diff_map[k,
                         max(i - RAD, 0): min(i + RAD + 1, diff_map[k].shape[0]),
                         max(j - RAD, 0): min(j + RAD + 1, diff_map[k].shape[1])] = \
                1 - diff_map[k,
                             max(i - RAD, 0): min(i + RAD + 1, diff_map[k].shape[0]),
                             max(j - RAD, 0): min(j + RAD + 1, diff_map[k].shape[1])]
                visit_map[max(i - RAD, 0): min(i + RAD + 1, diff_map[k].shape[0]),
                          max(j - RAD, 0): min(j + RAD + 1, diff_map[k].shape[1])] = 1

test_unchanged_ori_context_hard = test_ori_context.copy()[diff_map != 0]
test_unchanged_steg_context_hard = test_steg_context.copy()[diff_map != 0]

idx = np.random.permutation(len(test_unchanged_ori_context_hard))
test_unchanged_ori_context_hard = test_unchanged_ori_context_hard[idx][:NUM_HARD]
idx = np.random.permutation(len(test_unchanged_steg_context_hard))
test_unchanged_steg_context_hard = test_unchanged_steg_context_hard[idx][:NUM_HARD]

f = open("ensemble_clf_PH4.pkl", "rb")
clf_list = pickle.load(f)
f.close()

# def Shrink(X, shrinkArg, max_pooling=True, padding=True):
#     if max_pooling:
#         X = block_reduce(X, (1, 2, 2, 1), np.max)
#     win = shrinkArg['win']
#     if padding:
#         new_X = []
#         for each in X:
#             each = cv2.copyMakeBorder(each, win // 2, win // 2, win // 2, win // 2, cv2.BORDER_REFLECT)
#             new_X.append(each)
#         new_X = np.array(new_X)[:, :, :, None]
#     else:
#         new_X = X
#     X = view_as_windows(new_X, (1, win, win, 1))
#     X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
#     return X
#
#
# # example callback function for how to concate features from different hops
# def Concat(X, concatArg):
#     return X
#
# f = open("PixelHopUniform_4PH.pkl", 'rb')
# p2 = pickle.load(f)
# f.close()
#
# counts = p2.counts
# for i in range(1, len(counts)):
#     counts[i] += counts[i - 1][-1]

ori_prob = []
steg_prob = []
for i in range(1, len(CHANNEL_RANGE)):
    print(i)
    test_ori_vec = test_unchanged_ori_context_hard[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    test_steg_vec = test_unchanged_steg_context_hard[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    clf = clf_list[i - 1]
    ori_prob.append(clf.predict_proba(test_ori_vec)[:, 1])
    steg_prob.append(clf.predict_proba(test_steg_vec)[:, 1])

ori_prob = np.array(ori_prob).transpose()
steg_prob = np.array(steg_prob).transpose()

f = open("final_XG_BOOST", "rb")
svm_best = pickle.load(f)
f.close()

print("DENOTING THE PERCENTAGE OF SELECTED POINTS")
print("HARD UNCHANGED ORI SCORE (COMPARED WITH TOTAL 0):", svm_best.score(ori_prob, 0 * np.ones(len(ori_prob))))
print("HARD UNCHANGED STEG SCORE (COMPARED WITH TOTAL 0):", svm_best.score(steg_prob, 0 * np.ones(len(steg_prob))))
