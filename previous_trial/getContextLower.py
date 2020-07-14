import numpy as np
import cv2
from skimage import io
import os
from framework.tree_PixelHop import PixelHop_Unit, PixelHop_trans
import _pickle as cPickle
from skimage.measure import block_reduce
from previous_trial.config_img_stag import config

save_dir_matFiles = config['save_dir_matFiles']
window_rad = config['wind_rad']


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


"""
test hop unchanged, 
the PixelHop_trans is changed, two output elements are deleted,
idx__stop_list is abandoned. 
Leaf_node_thres is changed.
"""


def test_one_hop(ext_test_imgs):
    test_one_hop_attri_saab = PixelHop_Unit(ext_test_imgs, dilate=1, window_size=window_rad,
                                            idx_list=None, getK=False, pad='reflect',
                                            weight_root=os.path.join(save_dir_matFiles, 'weight'),
                                            Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
                                            PCA_ener_percent=0.99,
                                            useDC=False, stride=None, getcov=0, split_spec=0, hopidx=1)
    test_one_hop_attri_biased = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
                                               feature_ori=test_one_hop_attri_saab,
                                               split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)
    return test_one_hop_attri_biased


def test_second_hop(test_one_hop_attri_response):
    test_sec_hop_attri_saab = PixelHop_Unit(test_one_hop_attri_response, dilate=1, window_size=window_rad,
                                            idx_list=None, getK=False, pad='reflect',
                                            weight_root=os.path.join(save_dir_matFiles, 'weight'),
                                            Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
                                            PCA_ener_percent=0.97,
                                            useDC=False, stride=None, getcov=0, split_spec=1, hopidx=2)
    # print("sec hop neighborhood construction shape:", test_sec_hop_attri_saab.shape)
    test_sec_hop_attri_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
                                                 feature_ori=test_sec_hop_attri_saab,split_spec=1,
                                                 hopidx=2, Pass_Ener_thrs=0.1)
    # print("sec hop attri shape:", test_sec_hop_attri_response.shape)
    return test_sec_hop_attri_response


def test_third_hop(test_sec_hop_attri_response):
    test_third_hop_attri_saab = PixelHop_Unit(test_sec_hop_attri_response, dilate=1, window_size=window_rad,
                                              idx_list=None, getK=False, pad='reflect',
                                              weight_root=os.path.join(save_dir_matFiles, 'weight'),
                                              Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
                                              PCA_ener_percent=0.97,
                                              useDC=False, stride=None, getcov=0, split_spec=1, hopidx=3)
    # print("Hop-3 feature ori shape:", test_third_hop_attri_saab.shape)
    test_third_hop_attri_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
                                                   feature_ori=test_third_hop_attri_saab,
                                                   split_spec=1, hopidx=3, Pass_Ener_thrs=0.1)
    # print("third hop attri shape:", test_third_hop_attri_response.shape)
    return test_third_hop_attri_response


def test_forth_hop(test_third_hop_attri_response):
    test_fourth_hop_attri_saab = PixelHop_Unit(test_third_hop_attri_response, dilate=1, window_size=window_rad,
                                               idx_list=None, getK=False, pad='reflect',
                                               weight_root=os.path.join(save_dir_matFiles, 'weight'),
                                               Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None,
                                               PCA_ener_percent=0.97,
                                               useDC=False, stride=None, getcov=0, split_spec=1, hopidx=4)
    # print("Hop-3 feature ori shape:", test_third_hop_attri_saab.shape)
    test_fourth_hop_attri_response = PixelHop_trans(weight_name=os.path.join(save_dir_matFiles, 'weight'),
                                                    feature_ori=test_fourth_hop_attri_saab,
                                                    split_spec=1, hopidx=4, Pass_Ener_thrs=0.1)
    # print("third hop attri shape:", test_third_hop_attri_response.shape)
    return test_fourth_hop_attri_response


def getContextFromImgs(imgs, weight_root):
    """
    get all the context for the subtree node
    attribute: attribute of the latter part.
    ori_imgs: input images, to get the size of context map.
    """
    one_res = test_one_hop(imgs)
    sec_input = block_reduce(one_res, block_size=(1, 2, 2, 1), func=np.max)

    fread = open(os.path.join(weight_root + str(1), 'idx_to_pass.pkl'), 'rb')
    pass_idx = cPickle.load(fread)
    fread.close()
    all_nodes = np.arange(one_res.shape[-1]).tolist()
    for i in pass_idx:
        if i in all_nodes:
            all_nodes.remove(i)
    leaf_idx = all_nodes
    one_res = one_res[:, :, :, leaf_idx]
    print("Hop-1 attri shape:", one_res.shape)

    sec_res = test_second_hop(sec_input)
    thi_input = block_reduce(sec_res, block_size=(1, 2, 2, 1), func=np.max)

    fread = open(os.path.join(weight_root + str(2), 'idx_to_pass.pkl'), 'rb')
    pass_idx = cPickle.load(fread)
    fread.close()
    all_nodes = np.arange(sec_res.shape[-1]).tolist()
    for i in pass_idx:
        if i in all_nodes:
            all_nodes.remove(i)
    leaf_idx = all_nodes
    sec_res = sec_res[:, :, :, leaf_idx]
    print("Hop-2 attri shape:", sec_res.shape)

    # Hop-3
    thi_res = test_third_hop(thi_input)
    for_input = block_reduce(thi_res, block_size=(1, 2, 2, 1), func=np.max)

    fread = open(os.path.join(weight_root + str(3), 'idx_to_pass.pkl'), 'rb')
    pass_idx = cPickle.load(fread)
    fread.close()
    all_nodes = np.arange(thi_res.shape[-1]).tolist()
    for i in pass_idx:
        if i in all_nodes:
            all_nodes.remove(i)
    leaf_idx = all_nodes
    thi_res = thi_res[:, :, :, leaf_idx]
    print("Hop-3 attri shape:", thi_res.shape)

    # Hop-4
    for_res = test_forth_hop(for_input)
    print("Hop-4 attri shape:", for_res.shape)

    return one_res, sec_res, thi_res, for_res


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


def sobelImgs(gray_images):
    gradients = []
    for i in range(len(gray_images)):
        gradients.append(sobel(gray_images[i]))
    gradients = np.array(gradients)
    return gradients



def sobel(gray_image):
    def gradNorm(grad):
        return 255 * (grad - np.min(grad)) / (np.max(grad) - np.min(grad))

    m, n = gray_image.shape[0], gray_image.shape[1]
    panel_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    panel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    X = cv2.filter2D(gray_image, -1, panel_X)
    Y = cv2.filter2D(gray_image, -1, panel_Y)
    grad = np.sqrt((X ** 2) + (Y ** 2))
    return gradNorm(grad).astype(int)


if __name__ == "__main__":
    """
    This program get context and selected location based on 
    high-rate stego images and original images.  
    """
    thres_bin_edges = config['thres_bin_edges']
    ori_image_dir = config['ori_image_dir']
    stego_image_high_dir = config['stego_image_dir']
    stego_image_low_dir = config['stego_image_test_dir']
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

    print("Begin preprocessing")
    _, stego_imgs_low = readImgToNumpy_pairs(ori_image_dir, stego_image_low_dir)

    diff_pos = np.load(os.path.join(save_dir_matFiles, context_dir, 'diff_location.npy'))
    same_pos = np.load(os.path.join(save_dir_matFiles, context_dir, 'same_location.npy'))

    stego_imgs_low = stego_imgs_low[:, :, :, None]

    weight_root = os.path.join(save_dir_matFiles, weight_root_name)

    stg_1_context, stg_2_context, stg_3_context, stg_4_context = getContextFromImgs(stego_imgs_low, weight_root)

    print("Finish preprocessing")

    print('Begin to select context')
    selected_steg_context_diff = []
    for each_pos in diff_pos:
        kk, ii, jj = each_pos
        stg_context = np.concatenate(
            (getDataFromIthHop(stg_1_context[kk], 1, ii, jj),
             getDataFromIthHop(stg_2_context[kk], 2, ii, jj),
             getDataFromIthHop(stg_3_context[kk], 3, ii, jj),
             getDataFromIthHop(stg_4_context[kk], 4, ii, jj)),
            axis=0
        )
        selected_steg_context_diff.append(stg_context)
    selected_steg_context_diff = np.array(selected_steg_context_diff)
    # changed point for stego and cover
    selected_steg_context_same = []
    for each_pos in same_pos:
        kk, ii, jj = each_pos
        stg_context = np.concatenate(
            (getDataFromIthHop(stg_1_context[kk], 1, ii, jj),
             getDataFromIthHop(stg_2_context[kk], 2, ii, jj),
             getDataFromIthHop(stg_3_context[kk], 3, ii, jj),
             getDataFromIthHop(stg_4_context[kk], 4, ii, jj)),
            axis=0
        )
        selected_steg_context_same.append(stg_context)
    selected_steg_context_same = np.array(selected_steg_context_same)
    np.save(os.path.join(save_dir_matFiles, context_dir, 'selected_stg_unchanged_context_0.005.npy'), selected_steg_context_same)
    np.save(os.path.join(save_dir_matFiles, context_dir, 'selected_stg_context_diff_0.005.npy'), selected_steg_context_diff)
    print('saved successfully')

