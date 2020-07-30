import os
import pickle
import numpy as np
import cv2
from pixelhop2 import Pixelhop2
from skimage.measure import block_reduce
from skimage.util import view_as_windows

#########################################################
# STEP 1: TRAIN PIXELHOP BY STEGO IMAGE AND COVER IMAGE #
#            TRAINED PIXELHOP UNIT IS SAVED             #
#########################################################
ori_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
ori_img_save_path = r"/mnt/zhengwen/new_trial/BOSSbase_reverse"
steg_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'
steg_img_save_path = r"/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05_reverse"

file_names = os.listdir(ori_img_path)
file_names.sort(key=lambda x: int(x[:-4]))

count = 0
for file_name in file_names:
    print(count)
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)
    new_ori_img = np.zeros(ori_img.shape, "uint8")
    for i in range(ori_img.shape[0]):
        for j in range(ori_img.shape[1]):
            new_ori_img[i, j] = int('{:08b}'.format(ori_img[i, j])[::-1], 2)
    cv2.imwrite(os.path.join(ori_img_save_path, file_name), new_ori_img)

    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)
    new_steg_img = np.zeros(steg_img.shape, "uint8")
    for i in range(steg_img.shape[0]):
        for j in range(steg_img.shape[1]):
            new_steg_img[i, j] = int('{:08b}'.format(steg_img[i, j])[::-1], 2)
    cv2.imwrite(os.path.join(steg_img_save_path, file_name), new_steg_img)

    count += 1
