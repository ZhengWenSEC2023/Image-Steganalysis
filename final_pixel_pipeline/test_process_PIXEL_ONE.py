import os
import pickle
import numpy as np
import cv2
from skimage.util import view_as_windows
from scipy.io import loadmat
P_TEST_NUM_TOTAL = 2000

PERCENTAGE = 0.1
NUM_POINTS = int(PERCENTAGE * 256 * 256)
print(P_TEST_NUM_TOTAL)
np.random.seed(23)
ori_test_img = []
steg_test_img = []
ori_img_path = r'/mnt/zhengwen/new_trial/test_BOSS_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/test_BOSS_S_UNIWARD_05'
ori_rho_path = r'/mnt/zhengwen/new_trial/test_BOSS_resize_SUNI_rho'
steg_rho_path = r'/mnt/zhengwen/new_trial/test_BOSS_S_UNIWARD_05_SUNI_rho'
file_names = os.listdir(ori_img_path)
file_names.sort(key=lambda x: int(x[:-4]))


# mi = np.array([0.00E+00, 0.00E+00, 0.00E+00, 7.00E-04, 5.42E-04,
#                0.00E+00, 1.54E-05, 0.00E+00, 2.81E-03, 1.27E-03,
#                2.15E-04, 0.00E+00, 4.79E-04, 0.00E+00, 0.00E+00,
#                0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00,
#                8.04E-04, 0.00E+00, 0.00E+00, 1.15E-03, 0.00E+00])

mi = np.array([0., 0., 0.0055546, 0.00269855, 0.0018994,
               0.00361661, 0.00322969, 0.00379988, 0.00401491, 0.00291028,
               0.00267892, 0.00260812, 0.00362729, 0.0046669, 0.00323584,
               0.00324821, 0.00388679, 0.00166108, 0.0043296, 0.00273997,
               0.00265786, 0.00212213, 0.00287289, 0.00278374, 0.])

selected = np.where(mi != 0)[0]


steg_patches = []
cover_patches = []
count = 0

for file_name in file_names:
    rho_file_name = file_name + '.mat'
    cur_steg_patches = []
    cur_cover_patches = []
    print(count)
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)[:, :, None]
    ori_rho = loadmat(os.path.join(ori_rho_path, rho_file_name))["rho"]
    ori_rho = ori_rho.reshape(-1)
    visit_ori = np.zeros(256 * 256)
    ori_idx = np.argsort(ori_rho, kind="quicksort")
    visit_ori[ori_idx[:NUM_POINTS]] = 1
    visit_ori = visit_ori.reshape((256, 256))
    visit_ori = visit_ori != 0

    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)[:, :, None]
    steg_rho = loadmat(os.path.join(steg_rho_path, rho_file_name))["rho"]
    steg_rho = steg_rho.reshape(-1)
    visit_steg = np.zeros(256 * 256)
    steg_idx = np.argsort(steg_rho, kind="quicksort")
    visit_steg[steg_idx[:NUM_POINTS]] = 1
    visit_steg = visit_steg.reshape((256, 256))
    visit_steg = visit_steg != 0

    # diff_map = np.squeeze(ori_img - steg_img)
    # for i in range(4, 252):
    #     for j in range(4, 252):
    #         if diff_map[i, j] != 0:
    #             cur_steg_patches.append(steg_img[i - 4: i + 5, j - 4: j + 5])
    #             cur_cover_patches.append(ori_img[i - 4: i + 5, j - 4: j + 5])
    for i in range(4, 252):
        for j in range(4, 252):
            if visit_ori[i, j] != 0:
                cur_steg_patches.append(steg_img[i - 4: i + 5, j - 4: j + 5])
            if visit_steg[i, j] != 0:
                cur_cover_patches.append(ori_img[i - 4: i + 5, j - 4: j + 5])
    count += 1
    steg_patches.append(np.array(cur_steg_patches))
    cover_patches.append(np.array(cur_cover_patches))
    if count == P_TEST_NUM_TOTAL:
        break


def Shrink(X, shrinkArg, max_pooling=True, padding=True):
    X = view_as_windows(X, (1, 5, 5, 1))
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
    return X

# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X


# set args for PixelHop++
SaabArgs = [{'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None}]
shrinkArgs = [{'func': Shrink, 'win': 5}]
concatArg = {'func': Concat}


# f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopStegoPts_5_5_1st_layer.pkl", 'rb')
f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopStegoPts_5_5_1st_layer_sampled.pkl", 'rb')

# f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PixelHopStegoPts_7_7_1st_layer.pkl", 'rb')
p2 = pickle.load(f)
f.close()

features_cover = []
features_steg = []

for cover_patch in cover_patches:
    features_cover.append(np.squeeze(p2.transform(cover_patch)[0])[:, :, :, selected])

for stego_patch in steg_patches:
    features_steg.append(np.squeeze(p2.transform(stego_patch)[0])[:, :, :, selected])

# shape should be (2000, 16, x, x, 25)

# # np.save("week_12_feature_cover_px1_PIXEL_TEST.npy", features_cover)
# # np.save("week_12_feature_steg_px1_PIXEL_TEST.npy", features_steg)
# np.save("week_12_feature_cover_px1_PIXEL_TEST_sample.npy", features_cover)
# np.save("week_12_feature_steg_px1_PIXEL_TEST_sample.npy", features_steg)
#
# selected_cover = features_cover[:, :, :, selected]
# selected_steg = features_steg[:, :, :, selected]
#
# # np.save("week_12_feature_cover_selected_px1_PIXEL_TEST.npy", selected_cover)
# # np.save("week_12_feature_steg_selected_px1_PIXEL_TEST.npy", selected_steg)
# np.save("week_12_feature_cover_selected_px1_PIXEL_TEST_sample.npy", selected_cover)
# np.save("week_12_feature_steg_selected_px1_PIXEL_TEST_sample.npy", selected_steg)

# f = open("image_wise_cover.pkl", "wb")
# pickle.dump(features_cover, f)
# f.close()
#
# f = open("image_wise_steg.pkl", "wb")
# pickle.dump(features_steg, f)
# f.close()

f = open("image_wise_cover_rho.pkl", "wb")
pickle.dump(features_cover, f)
f.close()

f = open("image_wise_steg_rho.pkl", "wb")
pickle.dump(features_steg, f)
f.close()
