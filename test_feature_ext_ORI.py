import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import os
from feature_ext_TWO import context_resize

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

f = open("PixelHopUniform.pkl", 'rb')
p2 = pickle.load(f)
f.close()

f_ori_test_img = []
f_steg_test_img = []

ori_img_path = r'/mnt/zhengwen/new_trial/test_BOSS_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/test_BOSS_S_UNIWARD_05'

file_names = os.listdir(ori_img_path)
file_names.sort(key=lambda x: int(x[:-4]))
count = 0
for file_name in file_names:
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)[:, :, None]
    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)[:, :, None]
    f_ori_test_img.append(ori_img)
    f_steg_test_img.append(steg_img)

# FEATURE EXTRACTION
f_ori_test_img = np.array(f_ori_test_img)
f_steg_test_img = np.array(f_steg_test_img)
diff_map = np.squeeze(f_ori_test_img - f_steg_test_img)

print("--- Testing ---")
f_ori_context = p2.transform(f_ori_test_img)
f_steg_context = p2.transform(f_steg_test_img)
counts = p2.counts

f_ori_context_1 = f_ori_context[0]
f_ori_context_2 = f_ori_context[1]
f_ori_context_3 = f_ori_context[2]
f_ori_context_2 = context_resize(f_ori_context_2)
f_ori_context_3 = context_resize(f_ori_context_3)

f_steg_context_1 = f_steg_context[0]
f_steg_context_2 = f_steg_context[1]
f_steg_context_3 = f_steg_context[2]
f_steg_context_2 = context_resize(f_steg_context_2)
f_steg_context_3 = context_resize(f_steg_context_3)

test_f_ori_context = np.concatenate((f_ori_context_1, f_ori_context_2, f_ori_context_3), axis=-1)
test_f_steg_context = np.concatenate((f_steg_context_1, f_steg_context_2, f_steg_context_3), axis=-1)

test_f_ori_vectors = test_f_ori_context[diff_map != 0]
test_f_steg_vectors = test_f_steg_context[diff_map != 0]

np.save("week8_test_ori_context_1.npy", f_ori_context_1)
np.save("week8_test_ori_context_2.npy", f_ori_context_2)
np.save("week8_test_ori_context_3.npy", f_ori_context_3)
np.save("week8_test_steg_context_1.npy", f_steg_context_1)
np.save("week8_test_steg_context_2.npy", f_steg_context_2)
np.save("week8_test_steg_context_3.npy", f_steg_context_3)

np.save("week8_test_ori_feature_vec.npy", test_f_ori_vectors)
np.save("week8_test_steg_feature_vec.npy", test_f_steg_vectors)