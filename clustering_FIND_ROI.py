import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import os

###############################################################################
#   STEP 2 PLUS: EXTRACT FEATURES BY TRAINED PIXELHOP UNIT FOR TEST IMAGES    #
#                EXTRACT FEATURES BY THE TRAINED PIXELHOP UNIT                #
###############################################################################


# F_TRAIN_NUM_TOTAL = 2500
# f = open("PixelHopUniform.pkl", 'rb')  # 3 PixelHop, win: 5, TH1:0.005, TH2:0.005, CH1: 15, CH2: 20, CH3: 25, TRAIN_TOTAL=500
f = open("PixelHopUniform_4PH.pkl", 'rb')
p2 = pickle.load(f)
f.close()

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

# FEATURE EXTRACTION
f_ori_test_img = np.array(f_ori_test_img)
f_steg_test_img = np.array(f_steg_test_img)
diff_map = np.squeeze(f_ori_test_img.astype("double") - f_steg_test_img.astype("double"))

f_ori_context = p2.transform(f_ori_test_img)
f_steg_context = p2.transform(f_steg_test_img)
counts = p2.counts

f_ori_context_1 = np.load("week8_test_ori_context_1_PH4.npy")
f_ori_context_2 = np.load("week8_test_ori_context_2_PH4.npy")
f_ori_context_3 = np.load("week8_test_ori_context_3_PH4.npy")
f_ori_context_4 = np.load("week8_test_ori_context_4_PH4.npy")
f_steg_context_1 = np.load("week8_test_steg_context_1_PH4.npy")
f_steg_context_2 = np.load("week8_test_steg_context_2_PH4.npy")
f_steg_context_3 = np.load("week8_test_steg_context_3_PH4.npy")
f_steg_context_4 = np.load("week8_test_steg_context_4_PH4.npy")

test_f_ori_vectors = test_f_ori_context[diff_map != 0]
test_f_steg_vectors = test_f_steg_context[diff_map != 0]

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

np.save("week8_test_ori_context_1_PH4.npy", f_ori_context_1)
np.save("week8_test_ori_context_2_PH4.npy", f_ori_context_2)
np.save("week8_test_ori_context_3_PH4.npy", f_ori_context_3)
np.save("week8_test_ori_context_4_PH4.npy", f_ori_context_4)
np.save("week8_test_steg_context_1_PH4.npy", f_steg_context_1)
np.save("week8_test_steg_context_2_PH4.npy", f_steg_context_2)
np.save("week8_test_steg_context_3_PH4.npy", f_steg_context_3)
np.save("week8_test_steg_context_4_PH4.npy", f_steg_context_4)
np.save("week8_diff_map_test_PH4.npy", diff_map)

np.save("week8_test_ori_feature_vec_PH4.npy", test_f_ori_vectors)
np.save("week8_test_steg_feature_vec_PH4.npy", test_f_steg_vectors)

# LOAD

# f_ori_context_1 = np.load("week8_test_ori_context_1.npy")
# f_ori_context_2 = np.load("week8_test_ori_context_2.npy")
# f_ori_context_3 = np.load("week8_test_ori_context_3.npy")
# f_steg_context_1 = np.load("week8_test_steg_context_1.npy")
# f_steg_context_2 = np.load("week8_test_steg_context_2.npy")
# f_steg_context_3 = np.load("week8_test_steg_context_3.npy")


