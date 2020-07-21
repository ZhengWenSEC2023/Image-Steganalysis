import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import os
import matplotlib.pyplot as plt
import seaborn as sns
#############################################################################################
# TEST SETP: FOR EACH DIFFERENT SET, USE THE TRAINED PIXELHOP, XGBOOST, GIVE THE PERCENTAGE #
#                        CHANGE THE PATH TO TEST DIFFERENT IMAGE SET                        #
#############################################################################################

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


print("BEGIN READ IMAGE")

f_ori_test_img = []
f_steg_test_img = []
ori_img_path = r'/mnt/zhengwen/new_trial/test_BOSS_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/test_BOSS_S_UNIWARD_05'
file_names = os.listdir(ori_img_path)
file_names.sort(key=lambda x: int(x[:-4]))
for file_name in file_names:
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)[:, :, None]
    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)[:, :, None]
    f_ori_test_img.append(ori_img)
    f_steg_test_img.append(steg_img)

f_ori_test_img = np.array(f_ori_test_img)
f_steg_test_img = np.array(f_steg_test_img)
# FEATURE EXTRACTION

print("BEGIN FEATURE EXTRACTION")
f = open("PixelHopUniform_4PH.pkl", 'rb')
p2 = pickle.load(f)
f.close()

f_ori_context = p2.transform(f_ori_test_img)
f_steg_context = p2.transform(f_steg_test_img)

f_ori_context_1 = f_ori_context[0]
f_ori_context_2 = f_ori_context[1]
f_ori_context_3 = f_ori_context[2]
f_ori_context_4 = f_ori_context[3]
f_ori_context_2 = context_resize(f_ori_context_2)
f_ori_context_3 = context_resize(f_ori_context_3)
f_ori_context_4 = context_resize(f_ori_context_4)

f_steg_context_1 = f_steg_context[0]
f_steg_context_2 = f_steg_context[1]
f_steg_context_3 = f_steg_context[2]
f_steg_context_4 = f_steg_context[3]
f_steg_context_2 = context_resize(f_steg_context_2)
f_steg_context_3 = context_resize(f_steg_context_3)
f_steg_context_4 = context_resize(f_steg_context_4)

test_f_ori_context = np.concatenate((f_ori_context_1, f_ori_context_2, f_ori_context_3, f_ori_context_4), axis=-1)
test_f_steg_context = np.concatenate((f_steg_context_1, f_steg_context_2, f_steg_context_3, f_steg_context_4), axis=-1)

test_f_ori_vectors = test_f_ori_context.reshape((-1, test_f_ori_context.shape[-1]))
test_f_steg_vectors = test_f_steg_context.reshape((-1, test_f_steg_context.shape[-1]))

# del f_ori_context, f_steg_context, test_f_ori_context, test_f_steg_context
# del f_ori_context_1, f_ori_context_2, f_ori_context_3, f_ori_context_4
# del f_steg_context_1, f_steg_context_2, f_steg_context_3, f_steg_context_4

print("BEGIN CALCULATE PROBABILITY FOR EACH PIXELHOP")

f = open("ensemble_clf_PH4.pkl", "rb")
clf_list = pickle.load(f)
f.close()

ori_prob = []
steg_prob = []
for i in range(1, len(CHANNEL_RANGE)):
    print(i)
    test_ori_vec = test_f_ori_vectors[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    test_steg_vec = test_f_steg_vectors[:, CHANNEL_RANGE[i - 1]: CHANNEL_RANGE[i]]
    clf = clf_list[i - 1]
    ori_prob.append(clf.predict_proba(test_ori_vec)[:, 1])
    steg_prob.append(clf.predict_proba(test_steg_vec)[:, 1])

ori_prob = np.array(ori_prob).transpose()
steg_prob = np.array(steg_prob).transpose()

print("BEGIN CALCULATE FINAL LABEL WITH XGBOOST")

f = open("final_XG_BOOST", "rb")
xgboost_final = pickle.load(f)
f.close()

ori_score = xgboost_final.score(ori_prob, 0 * np.ones(len(ori_prob)))
ori_label = xgboost_final.predict(ori_prob)
steg_score = xgboost_final.score(steg_prob, 0 * np.ones(len(steg_prob)))
steg_label = xgboost_final.predict(steg_prob)

percentage_ori = []
percentage_steg = []
for i in range(100):
    percentage_ori.append(np.sum(ori_label[i * 256 * 256: (i + 1) * 256 * 256] == 0) / (256*256))
    percentage_steg.append(np.sum(steg_label[i * 256 * 256: (i + 1) * 256 * 256] == 0) / (256*256))


plt.figure()
sns.distplot(percentage_ori, kde=False, label="Cover")
sns.distplot(percentage_steg, kde=False, label="Stego")
plt.title("Percentage of Points Labeled as 0 in a Whole Image")
plt.xlabel("Percentage of Points Labeled as 0")
plt.ylabel("Number of Images")
plt.legend()
plt.savefig(r"Hist of image label, (S-UNIWARD, payload=0_5)")

print("FINISHED")