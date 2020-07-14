import numpy as np
import cv2
import os
import h5py
from skimage.measure import block_reduce
from previous_trial.config_img_stag import config
from previous_trial.image_stag_train_freq import (test_one_hop, test_second_hop, test_third_hop, test_forth_hop,
                                                  readImgToNumpy, RobertsAlogrithm, binarized, getContext)
import pickle

print("--------TEST PROCESS--------")

thres_bin_edges = config['thres_bin_edges']
test_dir = config['test_dir']
save_dir_matFiles = config['save_dir_matFiles']
context_dir = config['context_dir']
weight_root_name = config['weight_root_name']
regressor_dir = config['regressor_dir']
regressor_name = config['regressor_name']
decision_thres = config['decision_thres']
test_result_dir = config['test_result_dir']
result_name = config['result_name']

# read test images
test_filenames = np.random.permutation(os.listdir(test_dir)).tolist()
test_imgs = readImgToNumpy(test_dir)
test_edges = RobertsAlogrithm(test_imgs)
bin_test_edges = binarized(test_edges, thres=thres_bin_edges)
test_edges = cv2.normalize(test_edges, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
print("test extended imgs shape:", test_imgs.shape)

pkl = open(os.path.join(save_dir_matFiles, regressor_dir, regressor_name), 'rb')
regr = pickle.load(pkl)
pkl.close()

result = []
for k in test_edges.shape[0]:
    test_img = test_imgs[k].copy()

    test_one_hop_attri_response = test_one_hop(test_img)
    print(test_one_hop_attri_response.shape)  # 154, 256, 384, 12
    test_one_hop_attri = test_one_hop_attri_response[:]  # 154, 256, 384, 8
    test_sec_hop_input = block_reduce(test_one_hop_attri_response, block_size=(1, 2, 2, 1),
                                      func=np.max)  # 154, 128, 192, 12
    # test Hop-2
    test_sec_pass, test_sec_stay, test_sec_hop_attri_response = test_second_hop(test_sec_hop_input,
                                                                                idx_stop_list=[])
    print(test_sec_pass.shape, test_sec_stay.shape, test_sec_hop_attri_response.shape)
    test_sec_hop_attri = test_sec_stay
    test_third_hop_input = block_reduce(test_sec_pass, block_size=(1, 2, 2, 1), func=np.max)
    # test Hop-3
    test_third_pass, test_third_stay, test_third_hop_attri_response = test_third_hop(test_third_hop_input,
                                                                                     idx_stop_list=[])
    print(test_third_pass.shape, test_third_stay.shape, test_third_hop_attri_response.shape)
    test_third_hop_attri = test_third_stay
    test_forth_hop_input = block_reduce(test_third_pass, block_size=(1, 2, 2, 1), func=np.max)
    # test Hop-4
    test_forth_hop_attri_response = test_forth_hop(test_forth_hop_input, idx_stop_list=[])
    test_forth_hop_attri = test_forth_hop_attri_response
    print(test_forth_hop_attri.shape)

    # concatenate to be context
    # test context 2
    test_context_2 = getContext(test_sec_hop_attri, test_edges)
    # test context 3
    test_context_3 = getContext(test_third_hop_attri, test_edges)
    # test context 4
    test_context_4 = getContext(test_forth_hop_attri, test_edges)
    test_context = np.concatenate((test_one_hop_attri, test_context_2, test_context_3, test_context_4), axis=-1)

    test_result = regr.predict(test_context)
    pos_percentage = np.sum(test_result > 0) / np.sum(test_result != 0)

    if pos_percentage < decision_thres:
        result.append(-1)
    else:
        result.append(1)

if not os.path.isdir(os.path.join(save_dir_matFiles, test_result_dir)):
    os.makedirs(os.path.join(save_dir_matFiles, test_result_dir))
h5f = h5py.File(os.path.join(save_dir_matFiles, test_result_dir, result_name), 'w')
h5f.create_dataset('result', data=result)
h5f.close()
