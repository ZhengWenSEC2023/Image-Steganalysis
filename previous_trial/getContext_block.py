import scipy.io as scio
import os
import numpy as np
from previous_trial.config_img_stag import config
import cv2

def readMatToNumpy(mat_dir):
    mat_names = os.listdir(mat_dir)
    mat_names.sort(key=lambda x: int(x[:-8]))
    mats = []
    for mat_name in mat_names:
        mat = scio.loadmat(os.path.join(mat_dir, mat_name))
        mats.append(mat['total_new'])
    mats = np.array(mats)
    return mats


def getDataFromIthHop(attribute_each_img, hop_index, i, j):
    """
    get context data from all hop instead of padding
    hop_index begin with 1
    e.g. hop 1 with (512, 512)
         hop 2 with (256, 256)
         the idx of hop 2 will be divided by 2
    """
    factor = 2 ** hop_index - 1
    new_i = i // factor
    new_j = j // factor
    return attribute_each_img[new_i, new_j].copy()


def getContext(attribute, hop_idx):
    """
    get all the context for the subtree node
    attribute: attribute of the latter part.
    ori_imgs: input images, to get the size of context map.
    """
    factor = 2 ** (hop_idx - 1)
    S = attribute.shape
    context_later = np.zeros((S[0], S[-1], factor * S[1], factor * S[2]))
    attribute = np.moveaxis(attribute, -1, 1)
    new_S = (attribute.shape[0], attribute.shape[1], factor * attribute.shape[2], factor * attribute.shape[3])
    for k in range(context_later.shape[0]):
        for i in range(context_later.shape[1]):
            context_later[k][i] = cv2.resize(attribute[k][i],
                                             (new_S[2], new_S[3]),
                                             interpolation=cv2.INTER_NEAREST)
    context_later = np.moveaxis(context_later, 1, -1)
    return context_later


def readMatToNumpy(mat_dir):
    mat_names = os.listdir(mat_dir)
    mat_names.sort(key=lambda x: int(x[:-13]))
    mats = []
    for mat_name in mat_names:
        mat = scio.loadmat(os.path.join(mat_dir, mat_name))
        mats.append(mat['total_new'])
    mats = np.array(mats)
    return mats


save_dir_matFiles = config['save_dir_matFiles']
window_rad = config['wind_rad']

if __name__ == "__main__":
    thres_bin_edges = config['thres_bin_edges']
    ori_image_dir = config['ori_image_dir']
    stego_image_dir = config['stego_image_dir']
    context_dir = config['context_dir']
    train_ori_context_name = config['train_ori_context_name']
    weight_root_name = config['weight_root_name']
    train_pos_target_name = config['train_pos_target_name']
    train_neg_target_name = config['train_neg_target_name']
    train_pos_contxt_name = config['train_pos_contxt_name']
    train_neg_contxt_name = config['train_neg_contxt_name']
    test_ori_contxt_name = config['test_ori_contxt_name']
    test_contxt_name = config['test_contxt_name']
    regressor_dir = config['regressor_dir']
    regressor_name = config['regressor_name']
    plot_dir = config['plot_dir']
    one_hop_stp_lst_name = config['one_hop_stp_lst_name']

    hop1_context = np.load("context_1_week6.npy")
    hop2_context = np.load("context_2_week6.npy")
    hop2_context = getContext(hop2_context, 2)
    hop3_context = np.load("context_3_week6.npy")
    hop3_context = getContext(hop3_context, 3)

    context = np.concatenate((hop1_context, hop2_context, hop3_context), axis=3)
    bland_wind = 2
    num_vec = 1000
    mats = readMatToNumpy('/mnt/zhengwen/image_steganalysis/train_prob_with_prob')

    positive_context_1 = []
    positive_context_2 = []
    positive_context_3 = []

    negative_context_1 = []
    negative_context_2 = []
    negative_context_3 = []

    for i in range(len(context)):
        print(i)

        cur_context_1 = hop1_context[i]
        cur_context_2 = hop2_context[i]
        cur_context_3 = hop3_context[i]
        cur_mat = mats[i]
        cur_max_mat = cur_mat.copy()
        cur_min_mat = cur_mat.copy()
        cur_num = num_vec

        while cur_num > 0:
            max_pos = np.where(cur_mat == np.max(cur_mat))
            max_i = max_pos[0][0]
            max_j = max_pos[1][0]
            cur_max_mat[max(max_i - bland_wind, 0): min(max_i + bland_wind, 512), max(max_j - bland_wind, 0): min(max_j + bland_wind, 512)] = 0

            positive_context_1.append(cur_context_1[max_i, max_j])
            positive_context_2.append(cur_context_2[max_i, max_j])
            positive_context_3.append(cur_context_3[max_i, max_j])

            min_pos = np.where(cur_mat == np.min(cur_mat))
            min_i = min_pos[0][0]
            min_j = min_pos[1][0]
            cur_min_mat[max(min_i - bland_wind, 0): min(min_i + bland_wind, 512), max(min_j - bland_wind, 0): min(min_j + bland_wind, 512)] = 1

            negative_context_1.append(cur_context_1[min_i, min_j])
            negative_context_2.append(cur_context_2[min_i, min_j])
            negative_context_3.append(cur_context_3[min_i, min_j])
            cur_num -= 1

    positive_context_1 = np.array(positive_context_1)
    positive_context_2 = np.array(positive_context_2)
    positive_context_3 = np.array(positive_context_3)
    negative_context_1 = np.array(negative_context_1)
    negative_context_2 = np.array(negative_context_2)
    negative_context_3 = np.array(negative_context_3)

    np.save('pos_context_1.npy', positive_context_1)
    np.save('pos_context_2.npy', positive_context_2)
    np.save('pos_context_3.npy', positive_context_3)
    np.save('neg_context_1.npy', negative_context_1)
    np.save('neg_context_2.npy', negative_context_2)
    np.save('neg_context_3.npy', negative_context_3)
