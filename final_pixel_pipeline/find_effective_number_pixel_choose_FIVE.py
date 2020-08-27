from scipy.io import loadmat
import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


NUM_TRAIN_FINAL = 3000

channels = [3, 7, 14, 20, 22, 23, 25, 26, 27, 33, 40, 46, 47, 48, 51, 52,
            58, 68, 71, 76, 77, 80]

print("BEGIN READ IMAGE")

f_ori_train_img = []
f_ori_train_rho = []
f_steg_train_img = []
f_steg_train_rho = []
ori_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'
ori_rho_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize_SUNI_rho'
steg_rho_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05_SUNI_rho'
img_file_names = os.listdir(ori_img_path)
img_file_names.sort(key=lambda x: int(x[:-4]))
count = 0
for img_file_name in img_file_names:
    rho_file_name = img_file_name + ".mat"
    ori_img = cv2.imread(os.path.join(ori_img_path, img_file_name), 0)[:, :, None]
    ori_rho = loadmat(os.path.join(ori_rho_path, rho_file_name))["rho"]
    steg_img = cv2.imread(os.path.join(steg_img_path, img_file_name), 0)[:, :, None]
    steg_rho = loadmat(os.path.join(steg_rho_path, rho_file_name))["rho"]

    f_ori_train_img.append(ori_img)
    f_ori_train_rho.append(ori_rho)
    f_steg_train_img.append(steg_img)
    f_steg_train_rho.append(steg_rho)
    count += 1
    print(count)
    if count == NUM_TRAIN_FINAL:
        break

f_ori_train_img = np.array(f_ori_train_img)
f_ori_train_rho = np.array(f_ori_train_rho)

f_steg_train_img = np.array(f_steg_train_img)
f_steg_train_rho = np.array(f_steg_train_rho)

percentage_stego_in_choosed_USING_COVER_RHO = []
percentage_choosed_stego_in_total_stego_USING_COVER_RHO = []
percentage_stego_in_choosed_USING_STEGO_RHO = []
percentage_choosed_stego_in_total_stego_USING_STEGO_RHO = []

count = 0
for PERCENTAGE in [0.001 * i for i in range(1, 500)]:
    NUM_POINTS = int(PERCENTAGE * 256 * 256)
    print(NUM_POINTS, count)
    count += 1

    temp_1 = 0
    temp_2 = 0
    temp_3 = 0
    temp_4 = 0

    for k in range(len(f_ori_train_rho)):

        # cover_rho
        cur_ori = f_ori_train_img[k]
        cur_steg = f_steg_train_img[k]
        cur_ori_rho = f_ori_train_rho[k]

        cur_diff = np.squeeze(cur_ori - cur_steg) != 0

        cur_ori_rho = cur_ori_rho.reshape(-1)

        cur_visit = np.zeros(256 * 256)
        cur_ori_idx = np.argsort(cur_ori_rho, kind="quicksort")
        cur_visit[cur_ori_idx[:NUM_POINTS]] = 1
        cur_visit = cur_visit.reshape((256, 256))

        cur_visit = cur_visit != 0

        temp_1 += np.sum(np.logical_and(cur_visit, cur_diff)) / np.sum(cur_visit)
        temp_2 += np.sum(np.logical_and(cur_visit, cur_diff)) / np.sum(cur_diff)

        # stego_rho
        cur_steg_rho = f_steg_train_rho[k]
        cur_steg_rho = cur_steg_rho.reshape(-1)
        cur_visit = np.zeros(256 * 256)
        cur_steg_idx = np.argsort(cur_steg_rho, kind="quicksort")
        cur_visit[cur_steg_idx[:NUM_POINTS]] = 1
        cur_visit = cur_visit.reshape((256, 256))
        cur_visit = cur_visit != 0
        temp_3 += np.sum(np.logical_and(cur_visit, cur_diff)) / np.sum(cur_visit)
        temp_4 += np.sum(np.logical_and(cur_visit, cur_diff)) / np.sum(cur_diff)

    percentage_stego_in_choosed_USING_COVER_RHO.append(temp_1 / len(f_ori_train_rho))
    percentage_choosed_stego_in_total_stego_USING_COVER_RHO.append(temp_2 / len(f_ori_train_rho))
    percentage_stego_in_choosed_USING_STEGO_RHO.append(temp_3 / len(f_ori_train_rho))
    percentage_choosed_stego_in_total_stego_USING_STEGO_RHO.append(temp_4 / len(f_ori_train_rho))

plt.figure()
plt.plot(percentage_stego_in_choosed_USING_STEGO_RHO, label="% stego in selected")
plt.plot(percentage_choosed_stego_in_total_stego_USING_STEGO_RHO, label="% stego in total")
plt.xlabel("% pixel chosen")
plt.ylabel("Percentage")
plt.title("curve using cost map of cover image (WOW)")
plt.legend()
plt.savefig("curve using cost map of stego image (WOW)")

# 1. (array([ 4,  5,  9, 10, 13, 20, 22, 39, 41, 45, 46, 47, 49, 56, 62, 63, 64,
#             65, 66, 70, 74, 79, 80]),)

# 2. (array([ 2,  4,  5, 13, 14, 15, 30, 39, 41, 43, 51, 52, 61, 62, 67, 70, 72,
#             73, 76, 78]),)

# 3. (array([ 2,  5,  6,  9, 14, 22, 24, 25, 29, 32, 34, 36, 37, 38, 39, 40, 43,
#             48, 50, 51, 60, 62, 66, 69, 71, 72, 73, 76, 77, 80]),)

# 4. (array([ 0,  1,  4,  5,  6,  9, 10, 15, 20, 22, 23, 32, 39, 42, 47, 51, 52,
#             54, 55, 56, 61, 62, 63, 64, 65, 67, 68, 71, 74, 77, 79, 80]),)
#
