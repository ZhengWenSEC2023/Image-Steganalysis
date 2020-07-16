import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import os
from sklearn.metrics import calinski_harabaz_score

###############################################################################
#   TEST STEP 0: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT FOR TEST IMAGES    #
#                EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT                #
###############################################################################

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


ROI_TRAIN_NUM = 1000
# f = open("PixelHopUniform.pkl", 'rb')  # 3 PixelHop, win: 5, TH1:0.005, TH2:0.005, CH1: 15, CH2: 20, CH3: 25, TRAIN_TOTAL=500
f = open("PixelHopUniform_4PH.pkl", 'rb')
p2 = pickle.load(f)
f.close()

f_ori_img = []
f_steg_img = []
ori_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'
file_names = os.listdir(ori_img_path)
file_names.sort(key=lambda x: int(x[:-4]))
count = 0
for file_name in file_names:
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)[:, :, None]
    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)[:, :, None]
    f_ori_img.append(ori_img)
    f_steg_img.append(steg_img)
    count += 1
    if count == ROI_TRAIN_NUM:
        break

# FEATURE EXTRACTION
f_ori_img = np.array(f_ori_img)
f_steg_img = np.array(f_steg_img)
diff_map = np.squeeze(f_ori_img.astype("double") - f_steg_img.astype("double"))

f_ori_context = p2.transform(f_ori_img)
f_steg_context = p2.transform(f_steg_img)
counts = p2.counts

f_ori_context_1 = f_ori_context[0]
f_ori_context_2 = f_ori_context[1]
f_ori_context_3 = f_ori_context[2]
f_ori_context_4 = f_ori_context[3]
f_ori_context_2 = context_resize(f_ori_context_2)
f_ori_context_3 = context_resize(f_ori_context_3)
f_ori_context_4 = context_resize(f_ori_context_4)
ori_context = np.concatenate((f_ori_context_1, f_ori_context_2, f_ori_context_3, f_ori_context_4), axis=-1)

del f_ori_context
del f_ori_context_1
del f_ori_context_2
del f_ori_context_3
del f_ori_context_4

f_steg_context_1 = f_steg_context[0]
f_steg_context_2 = f_steg_context[1]
f_steg_context_3 = f_steg_context[2]
f_steg_context_4 = f_steg_context[3]
f_steg_context_2 = context_resize(f_steg_context_2)
f_steg_context_3 = context_resize(f_steg_context_3)
f_steg_context_4 = context_resize(f_steg_context_4)
steg_context = np.concatenate((f_steg_context_1, f_steg_context_2, f_steg_context_3, f_steg_context_4), axis=-1)

del f_steg_context
del f_steg_context_1
del f_steg_context_2
del f_steg_context_3
del f_steg_context_4

NUM_CHANGED_POINTS = 500000
NUM_UNCHANGED_POINTS = 4500000

unchanged_context = np.concatenate((ori_context[diff_map != 0], steg_context[diff_map != 0]), axis=0)
np.random.shuffle(unchanged_context)
unchanged_context = unchanged_context[:NUM_CHANGED_POINTS]
changed_context = np.concatenate((ori_context[diff_map == 0], steg_context[diff_map == 0]), axis=0)[:NUM_UNCHANGED_POINTS]
np.random.shuffle(changed_context)
changed_context = changed_context[:NUM_UNCHANGED_POINTS]

del ori_context
del steg_context

context = np.concatenate((unchanged_context, changed_context), axis=0)
context_label = np.concatenate((0 * np.ones(len(unchanged_context)), 1 * np.ones(len(changed_context))))

np.save("unshuffled_clustering_context.npy", context)
np.save("unshuffled_clustering_label.npy", context_label)

print("FINISHED")
