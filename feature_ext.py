import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import os

#########################################################
#   STEP 2: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT   #
#     EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT     #
#########################################################

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


def diff_sample(diff_maps):
    win = 10
    new_diff_maps = []
    for diff_map in diff_maps:
        visit_map = np.zeros(diff_map.shape)
        for i in range(diff_map.shape[0]):
            for j in range(diff_map.shape[1]):
                if visit_map[i, j] == 0:
                    if diff_map[i, j] != 0:
                        diff_map[max(0, i - win): min(i + win + 1, 256), max(0, j - win): min(j + win + 1, 256)] = 0
                        diff_map[i, j] = 1
                        visit_map[max(0, i - win): min(i + win + 1, 256), max(0, j - win): min(j + win + 1, 256)] = 1
                    else:
                        continue
        new_diff_maps.append(diff_map)
    return np.array(new_diff_maps)


F_TRAIN_NUM_TOTAL = 2500
# f = open("PixelHopUniform.pkl", 'rb')  # 3 PixelHop, win: 5, TH1:0.005, TH2:0.005, CH1: 15, CH2: 20, CH3: 25, TRAIN_TOTAL=500
f = open("PixelHopUniform_4PH.pkl", 'rb')
p2 = pickle.load(f)
f.close()

f_ori_train_img = []
f_steg_train_img = []
ori_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'
file_names = os.listdir(ori_img_path)
file_names.sort(key=lambda x: int(x[:-4]))
count = 0
for file_name in file_names:
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)[:, :, None]
    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)[:, :, None]
    f_ori_train_img.append(ori_img)
    f_steg_train_img.append(steg_img)
    count += 1
    if count == F_TRAIN_NUM_TOTAL:
        break
# FEATURE EXTRACTION
f_ori_train_img = np.array(f_ori_train_img)
f_steg_train_img = np.array(f_steg_train_img)
diff_map = np.squeeze(f_ori_train_img.astype("double") - f_steg_train_img.astype("double"))

f_ori_context = p2.transform(f_ori_train_img)
f_steg_context = p2.transform(f_steg_train_img)
counts = p2.counts

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

train_f_ori_context = np.concatenate((f_ori_context_1, f_ori_context_2, f_ori_context_3, f_ori_context_4), axis=-1)
train_f_steg_context = np.concatenate((f_steg_context_1, f_steg_context_2, f_steg_context_3, f_steg_context_4), axis=-1)

diff_map = diff_sample(diff_map)

train_f_ori_vectors = train_f_ori_context[diff_map != 0]
train_f_steg_vectors = train_f_steg_context[diff_map != 0]

# SAVE

# np.save("week8_train_ori_context_1.npy", f_ori_context_1)
# np.save("week8_train_ori_context_2.npy", f_ori_context_2)
# np.save("week8_train_ori_context_3.npy", f_ori_context_3)
# np.save("week8_train_steg_context_1.npy", f_steg_context_1)
# np.save("week8_train_steg_context_2.npy", f_steg_context_2)
# np.save("week8_train_steg_context_3.npy", f_steg_context_3)
# np.save("week8_diff_map_train.npy", diff_map)
#
# np.save("week8_train_ori_feature_vec.npy", train_f_ori_vectors)
# np.save("week8_train_steg_feature_vec.npy", train_f_steg_vectors)

np.save("week8_train_ori_context_1_PH4.npy", f_ori_context_1)
np.save("week8_train_ori_context_2_PH4.npy", f_ori_context_2)
np.save("week8_train_ori_context_3_PH4.npy", f_ori_context_3)
np.save("week8_train_ori_context_4_PH4.npy", f_ori_context_4)
np.save("week8_train_steg_context_1_PH4.npy", f_steg_context_1)
np.save("week8_train_steg_context_2_PH4.npy", f_steg_context_2)
np.save("week8_train_steg_context_3_PH4.npy", f_steg_context_3)
np.save("week8_train_steg_context_4_PH4.npy", f_steg_context_4)
np.save("week8_diff_map_train_PH4.npy", diff_map)

np.save("week8_train_ori_feature_vec_PH4.npy", train_f_ori_vectors)
np.save("week8_train_steg_feature_vec_PH4.npy", train_f_steg_vectors)

# LOAD

# f_ori_context_1 = np.load("week8_train_ori_context_1.npy")
# f_ori_context_2 = np.load("week8_train_ori_context_2.npy")
# f_ori_context_3 = np.load("week8_train_ori_context_3.npy")
# f_steg_context_1 = np.load("week8_train_steg_context_1.npy")
# f_steg_context_2 = np.load("week8_train_steg_context_2.npy")
# f_steg_context_3 = np.load("week8_train_steg_context_3.npy")

# f_ori_context_1 = np.load("week8_train_ori_context_1_PH4.npy")
# f_ori_context_2 = np.load("week8_train_ori_context_2_PH4.npy")
# f_ori_context_3 = np.load("week8_train_ori_context_3_PH4.npy")
# f_ori_context_4 = np.load("week8_train_ori_context_4_PH4.npy")
# f_steg_context_1 = np.load("week8_train_steg_context_1_PH4.npy")
# f_steg_context_2 = np.load("week8_train_steg_context_2_PH4.npy")
# f_steg_context_3 = np.load("week8_train_steg_context_3_PH4.npy")
# f_steg_context_4 = np.load("week8_train_steg_context_4_PH4.npy")
