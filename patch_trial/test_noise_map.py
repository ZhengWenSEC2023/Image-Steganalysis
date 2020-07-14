import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def truncate_2(x):
    neg = ((x + 2) + abs(x + 2)) / 2 - 2
    return -(-neg + 2 + abs(- neg + 2)) / 2 + 2


def NoiseMap(image):
    q = [4.0, 12.0, 2.0]
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]

    filter_res1 = cv2.filter2D(image, -1, filter1, borderType=cv2.BORDER_REFLECT)
    filter_res2 = cv2.filter2D(image, -1, filter2, borderType=cv2.BORDER_REFLECT)
    filter_res3 = cv2.filter2D(image, -1, filter3, borderType=cv2.BORDER_REFLECT)

    tr_1 = truncate_2(filter_res1)
    tr_2 = truncate_2(filter_res2)
    tr_3 = truncate_2(filter_res3)

    res = np.array([tr_1, tr_2, tr_3])
    res = np.round(res)
    res[res > 2] = 2
    res[res < -2] = -2

    res = np.moveaxis(res, 0, -1)

    return res


if __name__ == '__main__':

    original_path = r'/mnt/zhengwen/new_trial/test_BOSS_resize'
    stego_path = r'/mnt/zhengwen/new_trial/test_BOSS_S_UNIWARD_05'

    image_names = os.listdir(original_path)
    image_names.sort(key=lambda x: int(x[:-4]))

    count = 0
    ori_images = []
    stego_images = []
    for image_name in image_names:
        print(count)
        ori_image = cv2.imread(os.path.join(original_path, image_name), 0)
        stego_image = cv2.imread(os.path.join(stego_path, image_name), 0)
        count += 1
        ori_images.append(ori_image)
        stego_images.append(stego_image)

        # if count == 200: break

    ori_images = np.array(ori_images).astype("double")
    stego_images = np.array(stego_images).astype("double")

    count = 0
    ori_noise_map = []
    for each_image in ori_images:
        ori_noise_map.append(NoiseMap(each_image))
        count += 1
        print(count)

    count = 0
    steg_noise_map = []
    for each_image in stego_images:
        steg_noise_map.append(NoiseMap(each_image))
        count += 1
        print(count)

    np.save(r"/mnt/zhengwen/new_trial/ori_noise_map_test.npy", ori_noise_map)
    np.save(r"/mnt/zhengwen/new_trial/steg_noise_map_test.npy", steg_noise_map)