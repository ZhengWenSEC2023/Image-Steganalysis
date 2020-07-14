#%%
import sys
import numpy as np
import pickle
import math
import cv2
from skimage import io
from sklearn.cluster import MiniBatchKMeans, KMeans
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import f1_score as f1
import warnings

warnings.filterwarnings('ignore')


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


if __name__=='__main__':

    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/multi_resolution/features/'

    # read in multi-resolution name_loc_prob pickle

    with open(save_dir_matFiles + 'original_resolution/ssl/name_loc_prob_rfw.pkl', 'rb') as fid:
        name_loc_prob_1 = pickle.load(fid)

    with open(save_dir_matFiles + '1-2_resolution/ssl/name_loc_prob.pkl', 'rb') as fid:
        name_loc_prob_2 = pickle.load(fid)

    with open(save_dir_matFiles + '1-4_resolution/lab/name_loc_prob_rfw.pkl', 'rb') as fid:
        name_loc_prob_4 = pickle.load(fid)


    # 1-1 resolution
    output_prob_map_1 = np.zeros((len(name_loc_prob_1), 256, 384))
    # gt_map = np.zeros((len(name_loc_prob_1), 256, 384))

    for k in range(len(name_loc_prob_1)):
        # name_loc_prob[k] is a dictionary
        splice_pixel_loc = name_loc_prob_1[k]['spliced_loc']  # list
        authen_pixel_loc = name_loc_prob_1[k]['authentic_loc']  # list
        for pos_pixel in range(len(splice_pixel_loc)):
            i = splice_pixel_loc[pos_pixel][0]
            j = splice_pixel_loc[pos_pixel][1]
            output_prob_map_1[k, i, j] = splice_pixel_loc[pos_pixel][-1]
            # gt_map[k, i, j] = 1

        for neg_pixel in range(len(authen_pixel_loc)):
            i = authen_pixel_loc[neg_pixel][0]
            j = authen_pixel_loc[neg_pixel][1]
            output_prob_map_1[k, i, j] = authen_pixel_loc[neg_pixel][-1]
            # gt_map[k, i, j] = -1


    # 1-2 resolution
    output_prob_map_2 = np.zeros((len(name_loc_prob_2), 128, 192))
    # gt_map = np.zeros((len(name_loc_prob), 256, 384))

    for k in range(len(name_loc_prob_2)):
        # name_loc_prob[k] is a dictionary
        splice_pixel_loc = name_loc_prob_2[k]['spliced_loc']  # list
        authen_pixel_loc = name_loc_prob_2[k]['authentic_loc']  # list
        for pos_pixel in range(len(splice_pixel_loc)):
            i = splice_pixel_loc[pos_pixel][0]
            j = splice_pixel_loc[pos_pixel][1]
            output_prob_map_2[k, i, j] = splice_pixel_loc[pos_pixel][-1]
            # gt_map[k, i, j] = 1

        for neg_pixel in range(len(authen_pixel_loc)):
            i = authen_pixel_loc[neg_pixel][0]
            j = authen_pixel_loc[neg_pixel][1]
            output_prob_map_2[k, i, j] = authen_pixel_loc[neg_pixel][-1]
            # gt_map[k, i, j] = -1



    # 1-4 resolution
    output_prob_map_4 = np.zeros((len(name_loc_prob_4), 64, 96))
    # gt_map = np.zeros((len(name_loc_prob), 256, 384))

    for k in range(len(name_loc_prob_4)):
        # name_loc_prob[k] is a dictionary
        splice_pixel_loc = name_loc_prob_4[k]['spliced_loc']  # list
        authen_pixel_loc = name_loc_prob_4[k]['authentic_loc']  # list
        for pos_pixel in range(len(splice_pixel_loc)):
            i = splice_pixel_loc[pos_pixel][0]
            j = splice_pixel_loc[pos_pixel][1]
            output_prob_map_4[k, i, j] = splice_pixel_loc[pos_pixel][-1]
            # gt_map[k, i, j] = 1

        for neg_pixel in range(len(authen_pixel_loc)):
            i = authen_pixel_loc[neg_pixel][0]
            j = authen_pixel_loc[neg_pixel][1]
            output_prob_map_4[k, i, j] = authen_pixel_loc[neg_pixel][-1]
            # gt_map[k, i, j] = -1


    # bilinear interpolation
    prob_map_2_resize = np.zeros((output_prob_map_1.shape))
    for k in range(output_prob_map_2.shape[0]):
        prob_map_2_resize[k] = cv2.resize(output_prob_map_2[k], (384, 256), interpolation=cv2.INTER_LINEAR)

    prob_map_4_resize = np.zeros((output_prob_map_1.shape))
    for k in range(output_prob_map_4.shape[0]):
        prob_map_4_resize[k] = cv2.resize(output_prob_map_4[k], (384, 256), interpolation=cv2.INTER_LINEAR)


    ensemble_12 = np.zeros((output_prob_map_1.shape))
    for k in range(output_prob_map_1.shape[0]):
        tmp = np.maximum(output_prob_map_1[k], prob_map_2_resize[k], prob_map_4_resize[k])
        tmp_pos = np.maximum(tmp, 0)


        ensemble_12[k] = tmp_pos


    for k in range(output_prob_map_1.shape[0]):
        plt.figure(0)
        plt.imshow(ensemble_12[k])
        plt.axis('off')

        plt.savefig(save_dir_matFiles + 'ensemble/' + name_loc_prob_1[k]['test_name'][:-4] + '_ensem_1and2and4.png')
        plt.close(0)


