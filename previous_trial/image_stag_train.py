import numpy as np
import cv2
from skimage import io
import scipy.io as scio
import os
from framework.tree_PixelHop import PixelHop_Unit, PixelHop_trans
import time
from skimage.measure import block_reduce
from previous_trial.config_img_stag import config
from skimage.util import view_as_windows

save_dir_matFiles = config['save_dir_matFiles']
window_rad = config['wind_rad']


def readImgToNumpy(image_dir):
    """
    Read images in a dir into a numpy array
    """
    img_names = os.listdir(image_dir)
    img_names.sort(key=lambda x: int(x[:-4]))
    imgs = []
    for img_name in img_names:
        img = io.imread(os.path.join(image_dir, img_name))
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs

def readMatToNumpy(mat_dir):
    mat_names = os.listdir(mat_dir)
    mat_names.sort(key=lambda x: int(x[:-8]))
    mats = []
    for mat_name in mat_names:
        mat = scio.loadmat(os.path.join(mat_dir, mat_name))
        mats.append(mat['total_new'])
    mats = np.array(mats)
    return mats

def readImgToNumpy_pairs(ori_dir, stego_dir):
    """
    read paired images sequentially, return ori_imgs and stego_imgs,
    image names based on ori_dir
    """
    img_names = os.listdir(ori_dir)
    ori_imgs = []
    stego_imgs = []
    for img_name in img_names:
        ori_img = io.imread(os.path.join(ori_dir, img_name))
        stego_img = io.imread(os.path.join(stego_dir, img_name))
        ori_imgs.append(ori_img)
        stego_imgs.append(stego_img)
    ori_imgs = np.array(ori_imgs)
    stego_imgs = np.array(stego_imgs)
    return ori_imgs, stego_imgs


def RobertsAlogrithm(imgs):  # change to structure edge
    """
    Return edges of a RGB image.
    """

    def RobertsOperator(roi):
        operator_first = np.array([[-1, 0], [0, 1]])
        operator_second = np.array([[0, -1], [1, 0]])
        return np.abs(np.sum(roi[1:, 1:] * operator_first)) + np.abs(np.sum(roi[1:, 1:] * operator_second))

    edges = []
    for i in range(imgs.shape[0]):
        img = cv2.copyMakeBorder(imgs[i], 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for j in range(1, img.shape[0]):
            for k in range(1, img.shape[1]):
                img[j, k] = RobertsOperator(img[j - 1: j + 2, k - 1: k + 2])
        edges.append(img[1:img.shape[0] - 1, 1:img.shape[1] - 1])
    edges = np.array(edges)
    return edges


def boundaryExtension(imgs, kernel_rad):
    """
    Mirror padding of a rep of img (n, w, h, c).
    kernel_rad: radius of the window.
    """
    extended_imgs = []
    for i in range(imgs.shape[0]):
        img = imgs[i].copy()
        extended_img = cv2.copyMakeBorder(img, kernel_rad, kernel_rad,
                                          kernel_rad, kernel_rad, cv2.BORDER_REFLECT)
        extended_imgs.append(extended_img)
    extended_imgs = np.array(extended_imgs)
    return extended_imgs


def binarized(edges, thres=20, dilate=False):
    """
    binarized(edges, thres=20, dilate=False)
    Change continuous edge maps to 0 and 1, if dilate, the edges will be thicker
    """
    edges = edges.copy()
    bin_edges = []
    for i in range(edges.shape[0]):
        edge = edges[i]
        edge[np.where(edge > thres)] = 255
        edge[np.where(edge <= thres)] = 0
        edge[np.where(edge == 255)] = 1
        bin_edges.append(edge)
    bin_edges = np.array(bin_edges)
    if not dilate:
        return bin_edges
    else:
        dil_edges = []
        kernel = np.ones((3, 3), np.uint8)
        for i in (bin_edges.shape[0]):
            dil_edge = bin_edges[i]
            dil_edge = cv2.dilate(dil_edge, kernel, iterations=1)
            dil_edges.append(dil_edge)
        dil_edges = np.array(dil_edges)
        return dil_edges


def Shrink(X):
    win = 5
    stride = 1
    new_X = []
    for each_map in X:
        new_X.append(cv2.copyMakeBorder(each_map, win // 2, win // 2, win // 2, win // 2, cv2.BORDER_REFLECT))
    X = np.array(new_X)
    if len(X.shape) == 3:
        X = X[:, :, :, None]
    X = view_as_windows(X, (1, win, win, 1), (1, stride, stride, 1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)[:, :, :, :, None]


def one_hop(response, test_response):
    print("---- Hop 1 ----")
    one_hop_attri_saab, all_saab = PixelHop_Unit(response, getK=True, idx_list=None,
                                                 dilate=2, window_size=window_rad, pad='reflect',
                                                 weight_root=os.path.join(save_dir_matFiles, 'weight'),
                                                 Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.6, num_kernels=None,
                                                 PCA_ener_percent=1,
                                                 useDC=False, stride=None, getcov=0, split_spec=0, hopidx=1)
    print("Hop-1 feature ori shape:", one_hop_attri_saab.shape)
    one_hop_attri_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
                                            feature_ori=one_hop_attri_saab,
                                            split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)

    test_response = Shrink(test_response)

    test_context = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
                                  feature_ori=test_response,
                                  split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)

    print("one hop attribute shape:", one_hop_attri_response.shape)
    # kernels = all_saab['leaf0'].tree['kernel']
    energy = all_saab['leaf0'].tree['ac_energy']
    energy = energy / np.sum(energy)
    # kernel_5 = np.array(
    #     [[0, 0, 0, 0, 0],
    #      [0, 1, 1, 1, 0],
    #      [0, 1, 1, 1, 0],
    #      [0, 1, 1, 1, 0],
    #      [0, 0, 0, 0, 0],
    #      ]
    # )
    # high_frez_pow = []
    #
    # thres_high = 0.7
    # engr_th_low = 0.02
    # for each_kernel in kernels:
    #     each_kernel = np.reshape(each_kernel, (5, 5))
    #     each_fft = np.fft.fftshift(np.fft.fft2(each_kernel))
    #     high_frez_pow.append(1 - np.sum(np.abs(each_fft * kernel_5)) / np.sum(np.abs(each_fft)))
    # high_frez_pow = np.array(high_frez_pow)
    # low_frez_channels = np.where((high_frez_pow < thres_high) & (energy > engr_th_low))

    engr_th = 0.02
    high_engr_channels = np.where(energy > engr_th)
    engr_th_low = 0.005
    second_high_channels = np.where(energy > engr_th_low)

    return (one_hop_attri_response[:, :, :, high_engr_channels], one_hop_attri_response[:, :, :, second_high_channels],
            energy[high_engr_channels], energy[second_high_channels],
            test_context[:, :, :, high_engr_channels], test_context[:, :, :, second_high_channels])


def sec_hop(response, energy_prev, test_input):
    print("---- Hop 2 ----")
    sec_hop_total = []
    sec_hop_energy = []
    sec_hop_test = []
    for c in range(response.shape[-1]):
        cur_response = response[:, :, :, c][:, :, :, None]
        cur_test = test_input[:, :, :, c][:, :, :, None]
        cur_attri_saab, cur_saab = PixelHop_Unit(cur_response, getK=True, idx_list=None,
                                                 dilate=2, window_size=window_rad, pad='reflect',
                                                 weight_root=os.path.join(save_dir_matFiles, 'weight'),
                                                 Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.6, num_kernels=None,
                                                 PCA_ener_percent=1,
                                                 useDC=False, stride=None, getcov=0, split_spec=0, hopidx=1)
        cur_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
                                      feature_ori=cur_attri_saab,
                                      split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)
        cur_test = Shrink(cur_test)
        cur_test_context = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
                                          feature_ori=cur_test,
                                          split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)

        print("sec hop attribute shape:", cur_response.shape)
        energy = cur_saab['leaf0'].tree['ac_energy']
        energy = energy / np.sum(energy) * energy_prev[c]

        if isinstance(sec_hop_total, list):
            sec_hop_total = cur_response
            sec_hop_energy = energy
            sec_hop_test = cur_test_context
        else:
            sec_hop_total = np.concatenate((sec_hop_total, cur_response), axis=3)
            sec_hop_energy = np.concatenate((sec_hop_energy, energy), axis=0)
            sec_hop_test = np.concatenate((sec_hop_test, cur_test_context), axis=3)

        # engr_th = 0.0005
        # high_engr_channels = np.where((energy > engr_th))
        #
        # to_append = np.squeeze(cur_response[:, :, :, high_engr_channels])
        # if len(to_append.shape) == 3:
        #     to_append = to_append[:, :, :, None]
        # if to_append.shape[-1] == 0:
        #     to_append = np.squeeze(cur_response[:, :, :, :3])
        # sec_hop_total.append(to_append.copy())
        # sec_hop_energy.append(energy[high_engr_channels].copy() if len(high_engr_channels) != 0 else energy[:3].copy())
        #
        # to_append_test = np.squeeze(cur_test_context[:, :, :, high_engr_channels])
        # if len(to_append_test.shape) == 3:
        #     to_append_test = to_append_test[:, :, :, None]
        # if to_append_test.shape[-1] == 0:
        #     to_append_test = np.squeeze(cur_test_context[:, :, :, :3])
        # sec_hop_test.append(to_append_test.copy())

        print("CURRENT C:", c)

    engr_th = 0.005
    high_engr_channels = np.where(sec_hop_energy > engr_th)
    engr_th_low = 0.002
    second_high_channels = np.where(sec_hop_energy > engr_th_low)

    return (sec_hop_total[:, :, :, high_engr_channels], sec_hop_total[:, :, :, second_high_channels],
            sec_hop_energy[high_engr_channels], sec_hop_energy[second_high_channels],
            sec_hop_test[:, :, :, high_engr_channels], sec_hop_test[:, :, :, second_high_channels])


def thr_hop(response, energy_prev, test_input):
    print("---- Hop 3 ----")
    sec_hop_total = []
    sec_hop_energy = []
    sec_hop_test = []
    for c in range(response.shape[-1]):
        cur_response = response[:, :, :, c][:, :, :, None]
        cur_test = test_input[:, :, :, c][:, :, :, None]
        cur_attri_saab, cur_saab = PixelHop_Unit(cur_response, getK=True, idx_list=None,
                                                 dilate=2, window_size=window_rad, pad='reflect',
                                                 weight_root=os.path.join(save_dir_matFiles, 'weight'),
                                                 Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.6, num_kernels=None,
                                                 PCA_ener_percent=1,
                                                 useDC=False, stride=None, getcov=0, split_spec=0, hopidx=1)
        cur_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
                                      feature_ori=cur_attri_saab,
                                      split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)
        cur_test = Shrink(cur_test)
        cur_test_context = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
                                          feature_ori=cur_test,
                                          split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)

        print("sec hop attribute shape:", cur_response.shape)
        energy = cur_saab['leaf0'].tree['ac_energy']
        energy = energy / np.sum(energy) * energy_prev[c]

        if isinstance(sec_hop_total, list):
            sec_hop_total = cur_response
            sec_hop_energy = energy
            sec_hop_test = cur_test_context
        else:
            sec_hop_total = np.concatenate((sec_hop_total, cur_response), axis=3)
            sec_hop_energy = np.concatenate((sec_hop_energy, energy), axis=0)
            sec_hop_test = np.concatenate((sec_hop_test, cur_test_context), axis=3)

        # engr_th = 0.0005
        # high_engr_channels = np.where((energy > engr_th))
        #
        # to_append = np.squeeze(cur_response[:, :, :, high_engr_channels])
        # if len(to_append.shape) == 3:
        #     to_append = to_append[:, :, :, None]
        # if to_append.shape[-1] == 0:
        #     to_append = np.squeeze(cur_response[:, :, :, :3])
        # sec_hop_total.append(to_append.copy())
        # sec_hop_energy.append(energy[high_engr_channels].copy() if len(high_engr_channels) != 0 else energy[:3].copy())
        #
        # to_append_test = np.squeeze(cur_test_context[:, :, :, high_engr_channels])
        # if len(to_append_test.shape) == 3:
        #     to_append_test = to_append_test[:, :, :, None]
        # if to_append_test.shape[-1] == 0:
        #     to_append_test = np.squeeze(cur_test_context[:, :, :, :3])
        # sec_hop_test.append(to_append_test.copy())

        print("CURRENT C:", c)

    engr_th = 0.008
    high_engr_channels = np.where(sec_hop_energy > engr_th)
    engr_th_low = 0.001
    second_high_channels = np.where(sec_hop_energy > engr_th_low)

    return (sec_hop_total[:, :, :, high_engr_channels], sec_hop_total[:, :, :, second_high_channels],
            sec_hop_energy[high_engr_channels], sec_hop_energy[second_high_channels],
            sec_hop_test[:, :, :, high_engr_channels], sec_hop_test[:, :, :, second_high_channels])


# def third_hop(response, energy_prev, test_input):
#     print("---- Hop 3 ----")
#     sec_hop_total = []
#     sec_hop_energy = []
#     sec_hop_test = []
#     for c in range(response.shape[-1]):
#         cur_response = response[:, :, :, c][:, :, :, None]
#         cur_test = test_input[:, :, :, c][:, :, :, None]
#         cur_attri_saab, cur_saab = PixelHop_Unit(cur_response, getK=True, idx_list=None,
#                                                  dilate=2, window_size=window_rad, pad='reflect',
#                                                  weight_root=os.path.join(save_dir_matFiles, 'weight'),
#                                                  Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.6, num_kernels=None,
#                                                  PCA_ener_percent=1,
#                                                  useDC=False, stride=None, getcov=0, split_spec=0, hopidx=1)
#         cur_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
#                                       feature_ori=cur_attri_saab,
#                                       split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)
#         cur_test = Shrink(cur_test)
#         cur_test_context = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
#                                           feature_ori=cur_test,
#                                           split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)
#
#         print("sec hop attribute shape:", cur_response.shape)
#         energy = cur_saab['leaf0'].tree['ac_energy']
#         energy = energy / np.sum(energy) * energy_prev[c]
#
#         engr_th = 0.0005
#         high_engr_channels = np.where((energy > engr_th))
#
#         to_append = np.squeeze(cur_response[:, :, :, high_engr_channels])
#         if len(to_append.shape) == 3:
#             to_append = to_append[:, :, :, None]
#         if to_append.shape[-1] == 0:
#             to_append = np.squeeze(cur_response[:, :, :, :3])
#         sec_hop_total.append(to_append.copy())
#         sec_hop_energy.append(energy[high_engr_channels].copy() if len(high_engr_channels) != 0 else energy[:3].copy())
#
#         to_append_test = np.squeeze(cur_test_context[:, :, :, high_engr_channels])
#         if len(to_append_test.shape) == 3:
#             to_append_test = to_append_test[:, :, :, None]
#         if to_append_test.shape[-1] == 0:
#             to_append_test = np.squeeze(cur_test_context[:, :, :, :3])
#         sec_hop_test.append(to_append_test.copy())
#
#         print("CURRENT C:", c)
#
#     sec_hop_total = np.concatenate(sec_hop_total, axis=3)
#     sec_hop_energy = np.concatenate(sec_hop_energy, axis=0)
#     sec_hop_test = np.concatenate(sec_hop_test, axis=3)
#     return sec_hop_total, sec_hop_energy, sec_hop_test


# """
# test hop unchanged,
# the PixelHop_trans is changed, two output elements are deleted,
# idx__stop_list is abandoned.
# Leaf_node_thres is changed.
# """
# def test_one_hop(ext_test_imgs):
#     test_one_hop_attri_saab = PixelHop_Unit(ext_test_imgs, dilate=1, window_size=window_rad,
#                                             idx_list=None, getK=False, pad='reflect',
#                                             weight_root=os.path.join(save_dir_matFiles, 'weight'),
#                                             Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
#                                             PCA_ener_percent=0.99,
#                                             useDC=False, stride=None, getcov=0, split_spec=0, hopidx=1)
#     _, _, test_one_hop_attri_biased = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
#                                                      feature_ori=test_one_hop_attri_saab,
#                                                      split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)
#     return test_one_hop_attri_biased
#
#
# def test_second_hop(test_one_hop_attri_response, idx_stop_list):
#     all_nodes = np.arange(test_one_hop_attri_response.shape[-1]).tolist()
#     for i in idx_stop_list:
#         if i in all_nodes:
#             all_nodes.remove(i)
#     intermediate_idx = all_nodes
#     test_sec_hop_attri_saab = PixelHop_Unit(test_one_hop_attri_response, dilate=1, window_size=window_rad,
#                                             idx_list=intermediate_idx, getK=False, pad='reflect',
#                                             weight_root=os.path.join(save_dir_matFiles, 'weight'),
#                                             Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
#                                             PCA_ener_percent=0.97,
#                                             useDC=False, stride=None, getcov=0, split_spec=1, hopidx=2)
#     # print("sec hop neighborhood construction shape:", test_sec_hop_attri_saab.shape)
#     test_sec_pass, test_sec_stay, test_sec_hop_attri_response = PixelHop_trans(
#         weight_name=os.path.join(save_dir_matFiles, 'weight'),
#         feature_ori=test_sec_hop_attri_saab,
#         split_spec=1, hopidx=2, Pass_Ener_thrs=0.1)
#     # print("sec hop attri shape:", test_sec_hop_attri_response.shape)
#     return test_sec_pass, test_sec_stay, test_sec_hop_attri_response
#
#
# def test_third_hop(test_sec_hop_attri_response, idx_stop_list):
#     all_nodes = np.arange(one_hop_attri_response.shape[-1]).tolist()
#     for i in idx_stop_list:
#         if i in all_nodes:
#             all_nodes.remove(i)
#     intermediate_idx = all_nodes
#     test_third_hop_attri_saab = PixelHop_Unit(test_sec_hop_attri_response, dilate=1, window_size=window_rad,
#                                               idx_list=intermediate_idx, getK=False, pad='reflect',
#                                               weight_root=os.path.join(save_dir_matFiles, 'weight'),
#                                               Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
#                                               PCA_ener_percent=0.97,
#                                               useDC=False, stride=None, getcov=0, split_spec=1, hopidx=3)
#     # print("Hop-3 feature ori shape:", test_third_hop_attri_saab.shape)
#     test_third_pass, test_third_stay, test_third_hop_attri_response = PixelHop_trans(
#         weight_name=os.path.join(save_dir_matFiles, 'weight'),
#         feature_ori=test_third_hop_attri_saab,
#         split_spec=1, hopidx=3, Pass_Ener_thrs=0.1)
#     # print("third hop attri shape:", test_third_hop_attri_response.shape)
#     return test_third_pass, test_third_stay, test_third_hop_attri_response
#
#
# def test_forth_hop(test_third_hop_attri_response, idx_stop_list):
#     all_nodes = np.arange(one_hop_attri_response.shape[-1]).tolist()
#     for i in idx_stop_list:
#         if i in all_nodes:
#             all_nodes.remove(i)
#     intermediate_idx = all_nodes
#     test_fourth_hop_attri_saab = PixelHop_Unit(test_third_hop_attri_response, dilate=1, window_size=window_rad,
#                                                idx_list=intermediate_idx, getK=False, pad='reflect',
#                                                weight_root=os.path.join(save_dir_matFiles, 'weight'),
#                                                Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
#                                                PCA_ener_percent=0.97,
#                                                useDC=False, stride=None, getcov=0, split_spec=1, hopidx=4)
#     # print("Hop-3 feature ori shape:", test_third_hop_attri_saab.shape)
#     _, _, test_fourth_hop_attri_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
#                                                           feature_ori=test_fourth_hop_attri_saab,
#                                                           split_spec=1, hopidx=4, Pass_Ener_thrs=0.1)
#     # print("third hop attri shape:", test_third_hop_attri_response.shape)
#     return test_fourth_hop_attri_response


def getContext(attribute, ori_imgs):
    """
    get all the context for the subtree node
    attribute: attribute of the latter part.
    ori_imgs: input images, to get the size of context map.
    """
    context_later = np.zeros((ori_imgs.shape[0], attribute.shape[-1], ori_imgs.shape[1], ori_imgs.shape[2]))
    attribute = np.moveaxis(attribute, -1, 1)
    for k in range(ori_imgs.shape[0]):
        for i in range(attribute.shape[1]):
            context_later[k][i] = cv2.resize(attribute[k][i],
                                             (ori_imgs.shape[1], ori_imgs.shape[2]),
                                             interpolation=cv2.INTER_NEAREST)
    context_later = np.moveaxis(context_later, 1, -1)
    return context_later


def getDataFromIthHop(attribute_each_img, hop_index, i, j):
    """
    get context data from all hop instead of padding
    e.g. hop 1 with (512, 512)
         hop 2 with (256, 256)
         the idx of hop 2 will be divided by 2
    """
    factor = 2 ** hop_index - 1
    new_i = i // factor
    new_j = j // factor
    return attribute_each_img[new_i, new_j].copy()


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

    test_imgs = readImgToNumpy('/mnt/zhengwen/image_steganalysis/test_img_with_prob')

    print("Begin preprocessing")
    # ori_imgs, stego_imgs = readImgToNumpy_pairs(ori_image_dir, stego_image_dir)
    # ori_imgs = ori_imgs[:, :, :, None]
    # stego_imgs = stego_imgs[:, :, :, None]
    # imgs = np.concatenate((ori_imgs, stego_imgs), axis=0)
    # num_ori_imgs = len(ori_imgs)
    # num_stego_imgs = len(stego_imgs)
    # labels = np.concatenate((np.ones(num_ori_imgs), -1 * np.ones(num_stego_imgs)), axis=0)
    # idx = np.random.permutation(labels.shape[0])
    # imgs = imgs[idx][:100]
    # labels = labels[idx][:100]
    #
    # diffs = ori_imgs.astype("double") - stego_imgs.astype("double")
    # print("Finish preprocessing")
    # edges = RobertsAlogrithm(imgs)
    # bin_edges = binarized(edges, thres=thres_bin_edges)
    # edges = cv2.normalize(edges, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    # target_values = edges * labels[:, None, None]
    # print("Finish preprocessing")
    imgs = readImgToNumpy(ori_image_dir)
    """
    Saab, feature extraction, all pixels
    """
    print(">>    FEATURE EXTRACTION     <<")
    weight_root = os.path.join(save_dir_matFiles, weight_root_name)
    start = time.time()
    # Hop-1
    one_inter_response, context_hop1, one_inter_engr, hop1_engr, test_one_inter_response, test_context_hop1 = one_hop(imgs, test_imgs)
    one_inter_response = np.squeeze(one_inter_response)
    context_hop1 = np.squeeze(context_hop1)
    test_context_hop1 = np.squeeze(test_context_hop1)
    test_one_inter_response = np.squeeze(test_one_inter_response)
    print("PixelHop 1 output", context_hop1.shape)

    # sec_hop_input = block_reduce(high_frez_response, block_size=(1, 2, 2, 1), func=np.max)
    sec_hop_input = block_reduce(one_inter_response, block_size=(1, 2, 2, 1), func=np.max)
    test_one_inter_response = block_reduce(test_one_inter_response, block_size=(1, 2, 2, 1), func=np.max)
    # Hop-2
    sec_inter_response, context_hop2, sec_inter_engr, hop2_engr, test_sec_inter_response, test_context_hop2 = sec_hop(sec_hop_input, one_inter_engr, test_one_inter_response)
    sec_inter_response = np.squeeze(sec_inter_response)
    context_hop2 = np.squeeze(context_hop2)
    test_sec_inter_response = np.squeeze(test_sec_inter_response)
    test_context_hop2 = np.squeeze(test_context_hop2)
    print("PixelHop 2 output", sec_inter_response.shape)
    # Hop-3
    thr_hop_input = block_reduce(sec_inter_response, block_size=(1, 2, 2, 1), func=np.max)
    test_sec_inter_response = block_reduce(test_sec_inter_response, block_size=(1, 2, 2, 1), func=np.max)
    thr_inter_response, context_hop3, thr_inter_engr, hop3_engr, test_thr_inter_response, test_context_hop3 = sec_hop(thr_hop_input, sec_inter_engr, test_sec_inter_response)
    thr_inter_response = np.squeeze(thr_inter_response)
    context_hop3 = np.squeeze(context_hop3)
    test_thr_inter_response = np.squeeze(test_thr_inter_response)
    test_context_hop3 = np.squeeze(test_context_hop3)



    print("Begin to save!")

    print(context_hop1.shape)
    print(context_hop2.shape)
    print(context_hop3.shape)
    print(one_inter_response.shape)
    print(sec_inter_response.shape)
    print(thr_inter_response.shape)

    np.save("context_1_week6.npy", context_hop1)
    np.save("context_2_week6.npy", context_hop2)
    np.save("context_3_week6.npy", context_hop3)
    np.save("test_context_1_week6.npy", test_context_hop1)
    np.save("test_context_2_week6.npy", test_context_hop2)
    np.save("test_context_3_week6.npy", test_context_hop3)

    # np.save("low_freq_hop1.npy", context_hop1)
    # np.save("high_freq_hop1.npy", one_inter_response)
    # np.save("high_low_freq_hop2.npy", sec_inter_response)
    #
    # np.save("low_freq_hop1_test.npy", test_context_hop1)
    # np.save("high_freq_hop1_test.npy", test_one_inter_response)
    # np.save("high_low_freq_hop2_test.npy", sec_test_context)

    # third_hop_input = block_reduce(sec_hop_attri_response, block_size=(1, 2, 2, 1), func=np.max)
    # # Hop-3
    # third_hop_attri_response = third_hop(third_hop_input)
    #
    # fread = open(weight_root + str(2) + '/idx_to_pass.pkl', 'rb')
    # pass_idx = cPickle.load(fread)
    # fread.close()
    #
    # all_nodes = np.arange(sec_hop_attri_response.shape[-1]).tolist()
    # for i in pass_idx:
    #     if i in all_nodes:
    #         all_nodes.remove(i)
    # leaf_idx = all_nodes
    #
    # sec_hop_attri = sec_hop_attri_response[:, :, :, leaf_idx]
    # print("Hop-2 attri shape:",sec_hop_attri.shape)
    #
    # fourth_hop_input = block_reduce(third_hop_attri_response, block_size=(1, 2, 2, 1), func=np.max)
    # # Hop-4
    # fourth_hop_attri_response = fourth_hop(fourth_hop_input)
    #
    # fread = open(weight_root + str(3) + '/idx_to_pass.pkl', 'rb')
    # pass_idx = cPickle.load(fread)
    # fread.close()
    #
    # all_nodes = np.arange(third_hop_attri_response.shape[-1]).tolist()
    # for i in pass_idx:
    #     if i in all_nodes:
    #         all_nodes.remove(i)
    # leaf_idx = all_nodes
    #
    # third_hop_attri = third_hop_attri_response[:, :, :, leaf_idx]
    # print("Hop-3 attri shape:", third_hop_attri.shape)
    #
    # fourth_hop_attri = fourth_hop_attri_response
    # print("Hop-4 attri shape:", fourth_hop_attri.shape)
    #
    # print(one_hop_attri.shape)
    # print(sec_hop_attri.shape)
    # print(third_hop_attri.shape)
    # print(fourth_hop_attri.shape)
    #
    # print("Pixel Hop training finished")
    #
    # """
    # calculate context
    # """
    #
    #
    #
    #
    # # # concatenate to be context
    # # context 1 is one_hop_attri
    # # context 2 is sec_hop_attri shape to 512*512
    # context_2 = getContext(sec_hop_attri, imgs)
    # context_3 = getContext(third_hop_attri, imgs)
    # context_4 = getContext(fourth_hop_attri, imgs)
    # context = np.concatenate((one_hop_attri, context_2, context_3, context_4), axis=-1)
    # # context finished
    #
    # # # write context to file
    # print("saving to files")
    # if not os.path.isdir(os.path.join(save_dir_matFiles, context_dir)):
    #     os.makedirs(os.path.join(save_dir_matFiles, context_dir))
    # h5f = h5py.File(os.path.join(save_dir_matFiles, context_dir, train_ori_context_name), 'w')
    # h5f.create_dataset('attri', data=context)
    # print("saved successfully")
    # # size_subset = 1000
    # # num_subset = int(context.shape[0] // size_subset + 1)
    # # for i in range(num_subset):
    # #     if np.size(context[i * size_subset: (i + 1) * size_subset]) != 0:
    # #         h5f.create_dataset('attribute' + str(i), data=context[i * size_subset: (i + 1) * size_subset])
    # h5f.close()
    #
    # # # read context from file
    # # context = []
    # # h5f = h5py.File(os.path.join(save_dir_matFiles, context_dir, train_ori_context_name), 'r')
    # # for i in range(len(list( h5f.keys() ))):
    # #     if isinstance(context, list):
    # #         context = h5f["attribute" + str(i)][:]
    # #     else:
    # #         context = np.concatenate( (context, h5f["attribute" + str(i)][:]), axis=0)
    # # h5f.close()
    # # print(context.shape)
    # end = time.time()
    # print("Time 1:", end - start)
    #
    # """
    # Select pixel features: positive
    # """
    # print(">>    SELECTING PIXEL FEATURES    <<")
    # print(context.shape)
    # pos_target = []
    # neg_target = []
    # pos_context = []
    # neg_context = []
    # counts_pos = np.zeros((edges.shape[0]), dtype=int)
    # counts_neg = np.zeros((edges.shape[0]), dtype=int)
    #
    # for k in range(edges.shape[0]):
    #     count = 0
    #     if labels[k] < 0:
    #         for i in range(edges.shape[1]):
    #             for j in range(edges.shape[2]):
    #                 if bin_edges[k, i, j] != 0:
    #                     neg_target.append(target_values[k, i, j])
    #                     neg_context.append(context[k, i, j])
    #                     count += 1
    #         counts_neg[k] = count
    #     else:
    #         for i in range(edges.shape[1]):
    #             for j in range(edges.shape[2]):
    #                 if bin_edges[k, i, j] != 0:
    #                     pos_target.append(target_values[k, i, j])
    #                     pos_context.append(context[k, i, j])
    #                     count += 1
    #         counts_pos[k] = count
    # print("total number of positive pixels:", counts_pos)
    # print("total number of negative pixels:", counts_neg)
    # print("pos target range:", np.max(pos_target), np.min(pos_target))
    # print("neg target range:", np.max(neg_target), np.min(neg_target))
    # print('saving target and context to files')
    # np.save(os.path.join(save_dir_matFiles, context_dir, train_pos_target_name), pos_target)
    # np.save(os.path.join(save_dir_matFiles, context_dir, train_neg_target_name), neg_target)
    # pos_context = np.array(pos_context)
    # neg_context = np.array(neg_context)
    # np.save(os.path.join(save_dir_matFiles, context_dir, train_pos_contxt_name), pos_context)
    # np.save(os.path.join(save_dir_matFiles, context_dir, train_neg_contxt_name), neg_context)
    # print('saved')
    # train_context = np.concatenate( (pos_context, neg_context), axis=0 )
    # train_target = np.concatenate( (pos_target, neg_target), axis=0 )

    # sum_counts_pos = sum(counts_pos)
    # sum_counts_neg = sum(counts_neg)
    # solve inbalancing, find equal and del, cost much time, CAREFUL
    # while sum_counts_pos > sum_counts_neg:
    #     for i in range(pos_context.shape[0]):
    #         for j in range(neg_context.shape[0]):
    #             if (pos_context[i] == neg_context[j]).all():
    #                 np.delete(pos_context, i, 0)
    #                 sum_counts_pos -= 1
    # while sum_counts_neg > sum_counts_pos:
    #     for i in range(neg_context.shape[0]):
    #         for j in range(pos_context.shape[0]):
    #             if (neg_context[i] == pos_context[j]).all():
    #                 np.delete(neg_context, i, 0)
    #                 sum_counts_neg -= 1

    # regr = RandomForestRegressor(n_estimators=200, random_state=0)
    # print('Begin to fit RF regressor')
    # regr.fit(train_context, train_target)
    # print('Fit finished')
    # if not os.path.isdir(os.path.join(save_dir_matFiles, regressor_dir)):
    #     os.makedirs(os.path.join(save_dir_matFiles, regressor_dir))
    # pkl = open(os.path.join(save_dir_matFiles, regressor_dir, regressor_name), 'wb')
    # pickle.dump(regr, pkl)
    # pkl.close()
    #
    # """
    # Val: percentage of positive class and negative class
    # """
    #
    # pos_result = regr.predict(pos_context)
    # neg_result = regr.predict(neg_context)
    #
    # pos_pos_percentage = np.sum(pos_result > 0) / np.sum(pos_result != 0)
    # pos_neg_percentage = np.sum(pos_result < 0) / np.sum(pos_result != 0)
    # neg_pos_percentage = np.sum(neg_result > 0) / np.sum(neg_result != 0)
    # neg_neg_percentage = np.sum(neg_result < 0) / np.sum(neg_result != 0)
    #
    # print("pos result", pos_pos_percentage, '%positive')
    # print("pos result", pos_neg_percentage, '%negative')
    # print("neg result", neg_pos_percentage, '%positive')
    # print("neg result", neg_neg_percentage, '%negative')
    #
    # if not os.path.isdir(os.path.join(save_dir_matFiles, plot_dir)):
    #     os.makedirs(os.path.join(save_dir_matFiles, plot_dir))
    # plt.hist(pos_result, bins=50, color=sns.desaturate("indianred", .8), alpha=.4)
    # plt.title("positive distribution")
    # plt.savefig(os.path.join(save_dir_matFiles, plot_dir, 'distribution_train_pos.png'))
    #
    # plt.hist(neg_result, bins=50, color=sns.desaturate("indianred", .8), alpha=.4)
    # plt.title("negative distribution")
    # plt.savefig(os.path.join(save_dir_matFiles, plot_dir, 'distribution_train_neg.png'))
    # """
    # Test
    # """
    # print("--------TEST PROCESS--------")
    #
    # test_dir = config['test_dir']
    # # read test images
    # test_filenames = np.random.permutation(os.listdir(test_dir)).tolist()
    # test_imgs = readImgToNumpy(test_dir)
    # test_edges = RobertsAlogrithm(test_imgs)
    # bin_test_edges = binarized(test_edges, thres=thres_bin_edges)
    # test_edges = cv2.normalize(test_edges, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    # print("test extended imgs shape:", test_imgs.shape)
    #
    # # %%
    # print(">>    FEARURE EXTRATION    <<")
    #
    # # extract context
    # # test hop-1
    # test_one_hop_attri_response = test_one_hop(test_imgs)
    # print(test_one_hop_attri_response.shape)  # 154, 256, 384, 12
    # test_one_hop_attri = test_one_hop_attri_response[:]  # 154, 256, 384, 8
    # test_sec_hop_input = block_reduce(test_one_hop_attri_response, block_size=(1, 2, 2, 1),
    #                                   func=np.max)  # 154, 128, 192, 12
    # # test Hop-2
    # test_sec_pass, test_sec_stay, test_sec_hop_attri_response = test_second_hop(test_sec_hop_input,
    #                                                                             idx_stop_list=[])
    # print(test_sec_pass.shape, test_sec_stay.shape, test_sec_hop_attri_response.shape)
    # test_sec_hop_attri = test_sec_stay
    # test_third_hop_input = block_reduce(test_sec_pass, block_size=(1, 2, 2, 1), func=np.max)
    # # test Hop-3
    # test_third_pass, test_third_stay, test_third_hop_attri_response = test_third_hop(test_third_hop_input,
    #                                                                                  idx_stop_list=[])
    # print(test_third_pass.shape, test_third_stay.shape, test_third_hop_attri_response.shape)
    # test_third_hop_attri = test_third_stay
    # test_forth_hop_input = block_reduce(test_third_pass, block_size=(1, 2, 2, 1), func=np.max)
    # # test Hop-4
    # test_forth_hop_attri_response = test_forth_hop(test_forth_hop_input, idx_stop_list=[])
    # test_forth_hop_attri = test_forth_hop_attri_response
    # print(test_forth_hop_attri.shape)
    #
    # # concatenate to be context
    # # test context 2
    # test_context_2 = getContext(test_sec_hop_attri, test_edges)
    # # test context 3
    # test_context_3 = getContext(test_third_hop_attri, test_edges)
    # # test context 4
    # test_context_4 = getContext(test_forth_hop_attri, test_edges)
    # test_context = np.concatenate((test_one_hop_attri, test_context_2, test_context_3, test_context_4), axis=-1)
    #
    # h5f = h5py.File(os.path.join(save_dir_matFiles, context_dir, test_ori_contxt_name), 'w')
    # size_subset = 1000
    # num_subset = int(test_context.shape[0] // size_subset + 1)
    # for i in range(num_subset):
    #     if np.size(test_context[i * size_subset: (i + 1) * size_subset]) != 0:
    #         h5f.create_dataset('attribute' + str(i), data=test_context[i * size_subset: (i + 1) * size_subset])
    # h5f.close()
    #
    # # test_context = []
    # # h5f = h5py.File( os.path.join(save_dir_matFiles, context_dir, test_ori_contxt_name), 'r')
    # # for i in range(len(list( h5f.keys() ))):
    # #     if isinstance(test_context, list):
    # #         test_context = h5f["attribute" + str(i)][:]
    # #     else:
    # #         test_context = np.concatenate( (test_context, h5f["attribute" + str(i)][:]), axis=0)
    # # h5f.close()
    # # print(test_context.shape)
    #
    # # %%  select positive & negative pixels of test imgs
    # selected_test_context = []
    # counts_test = np.zeros((test_edges.shape[0]), dtype=int)
    # for k in range(test_edges.shape[0]):
    #     count = 0
    #     for i in range(test_edges.shape[1]):
    #         for j in range(test_edges.shape[2]):
    #             if bin_edges[k, i, j] != 0:
    #                 selected_test_context.append(context[k, i, j])
    #                 count += 1
    #     counts_neg[k] = count
    # counts_test = np.array(counts_test)
    # np.save(os.path.join(save_dir_matFiles, context_dir, test_contxt_name), counts_test)
