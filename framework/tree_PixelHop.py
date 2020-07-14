# Alex
# Modified by Yijing

# yifanwang0916@outlook.com
# PixelHop unit

# feature: <4-D array>, (N, H, W, D)
# dilate: <int> dilate for pixelhop (default: 1)
# num_AC_kernels: <int> AC kernels used for Saab (default: 6)
# pad: <'reflect' or 'none' or 'zeros'> padding method (default: 'reflect)
# weight_name: <string> weight file (in '../weight/'+weight_name) to be saved or loaded. 
# getK: <bool> 0: using saab to get weight; 1: loaded pre-achieved weight
# useDC: <bool> add a DC kernel. 0: not use (out kernel is num_AC_kernels); 1: use (out kernel is num_AC_kernels+1)

# return <4-D array>, (N, H_new, W_new, D_new)

# PCA_ener_percent and num_kernels can't be None at the same time
# PCA_ener_percent = None, num_kernels = number
# PCA_ener_percent = %, num_kernels is ignored

import numpy as np
import pickle
import time
import math
import os
from skimage.util.shape import view_as_windows
from framework.saab import Saab
from framework.tree_Saab import treeSaab
import matplotlib.pyplot as plt
import cv2


def window_process(samples, kernel_size, stride):
    '''
    Create patches
    :param samples: [num_samples, feature_height, feature_width, feature_channel]
    :param kernel_size: int i.e. patch size
    :param stride: int
    :return patches: flattened, [num_samples, output_h, output_w, feature_channel*kernel_size^2]
    '''
    samples = np.pad(samples, (
        (0, 0), (int(kernel_size / 2), int(kernel_size / 2)), (int(kernel_size / 2), int(kernel_size / 2)), (0, 0)),
                     'reflect')
    n, h, w, c = samples.shape
    output_h = (h - kernel_size) // stride + 1
    output_w = (w - kernel_size) // stride + 1
    patches = view_as_windows(samples, (1, kernel_size, kernel_size, c), step=(1, stride, stride, c))
    patches = patches.reshape(n, output_h, output_w, kernel_size * kernel_size, c)
    return patches


def PixelHop_Neighbour(feature, dilate, window_size, pad):
    # feature: input feature, shape: n, x, y, channel (3/1)
    #  dilate: dilate number, only when window size=3, dilate
    #  window_size: 5 or 3, if == 5, no dilation, if == 3, use dilated window
    # print("------------------- Start: PixelHop_Neighbour")
    # S = feature.shape
    #
    # if pad == 'reflect':
    #     feature = np.pad(feature, (
    #         (0, 0), (window_size * dilate, window_size * dilate), (window_size * dilate, window_size * dilate), (0, 0)),
    #                      'reflect')
    # elif pad == 'zeros':
    #     feature = np.pad(feature, (
    #         (0, 0), (window_size * dilate, window_size * dilate), (window_size * dilate, window_size * dilate), (0, 0)),
    #                      'constant', constant_values=0)
    #
    # res = np.zeros((S[1], S[2], S[0], (2 * window_size + 1) ** 2, S[3]))
    #
    # feature = np.moveaxis(feature, 0, 2)  # 256, 384, 100, 3
    #
    # idx = (np.arange(2 * window_size + 1) - window_size) * dilate
    # for i in range(window_size * dilate, feature.shape[0] - window_size * dilate):
    #     for j in range(window_size * dilate, feature.shape[1] - window_size * dilate):
    #         tmp = []
    #         for ii in idx:
    #             for jj in idx:
    #                 if ii == 0 and jj == 0:
    #                     continue
    #                 iii = i + ii
    #                 jjj = j + jj
    #                 tmp.append(feature[iii, jjj])
    #         tmp.append(feature[i, j])
    #         tmp = np.array(tmp)  # 9, 100, 3
    #         tmp = np.moveaxis(tmp, 0, 1)  # 100, 9, 3
    #         res[i - window_size * dilate, j - window_size * dilate] = tmp  # 256, 384, 100, 9, 3
    #
    # res = np.moveaxis(res, 2, 0)  # 100, 256, 384, 9, 3
    # print("   <Neighborhood Info>Output feature shape: %s" % str(res.shape))
    # # print("------------------- End: PixelHop_Neighbour -> using %10f seconds"%(time.time()-t0))
    if pad == 0:
        flag = cv2.BORDER_REFLECT
    else:
        flag = cv2.BORDER_CONSTANT
    win = 2 * window_size + 1
    stride = dilate
    new_X = []
    for each_map in feature:
        new_X.append(cv2.copyMakeBorder(each_map, win // 2, win // 2, win // 2, win // 2, flag))
    X = np.array(new_X)
    if len(X.shape) == 3:
        X = X[:, :, :, None]
    X = view_as_windows(X, (1, win, win, 1), (1, stride, stride, 1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1, X.shape[3])


def PixelHop_Unit(feature, getK=True, idx_list=None, dilate=0, window_size=2, pad='reflect', weight_root='./tmp',
                  Pass_Ener_thrs=0.4, Leaf_Ener_thrs=0.001, num_kernels=None, PCA_ener_percent=None,
                  useDC=True, stride=None, getcov=0, split_spec=1, hopidx=1):
    '''
    Pass_Ener_thrs: the threshold to select the channel for passing down
    PCA_ener_percent: energy for PCA
    Leaf_Ener_thrs: the threshold for the leaf termination energy

    getK = 1: train pixelhop unit and save it into a dir, return the neighborhood map
    getK = 0: test and change the feature map to neighborhood map channelwisely
    '''
    print("=========== Create weight root ...")
    if not os.path.isdir(weight_root + str(hopidx)):
        os.makedirs(weight_root + str(hopidx))
    print("=========== Start: PixelHop_Unit")
    t0 = time.time()
    if getK == True:
        print(">>>>>====== Training ======<<<<<")
        # indicator = sample_indicator(gt)
        total_leaf_num = 0.0
        all_subTree = {}
        leaf_energy = []

        if split_spec == 1:
            """
            Channel-wise
            """
            if hopidx > 1:  # start from 2hop
                """
                Select several previous feature map to pass
                """
                fr = open(os.path.join(weight_root + str(hopidx - 1), 'leaf_energy.pkl'), 'rb')
                leaf_energy_last = pickle.load(fr)
                fr.close()

                not_end_leaf_idx = np.where(leaf_energy_last < Leaf_Ener_thrs)[
                    0].tolist()  # larger than leaf_ener_thrs, chose as intermediate node

                idx_list = not_end_leaf_idx  # fill in manually
                fwrite = open(os.path.join(weight_root + str(hopidx - 1), 'idx_to_pass.pkl'), 'wb')
                pickle.dump(idx_list, fwrite, protocol=2)
                fwrite.close()
            else:  # 1hop
                """
                All pass
                """
                idx_list = np.arange(0, feature.shape[-1]).tolist()  # 1,3,5,6,7...
            print(idx_list)
            print("<===================== Pass {} previous leaves to this Hop ======================>".format(
                len(idx_list)))

            feature_ori = np.zeros((feature.shape[0], feature.shape[1], feature.shape[2], (2 * window_size + 1) ** 2,
                                    feature.shape[-1]))  # 10,256,384,25,28
            for c in idx_list:
                # feature_pool = PixelHop_batchSample(feature[:,:,:,[c]], indicator, dilate, pad) # (n,spacial,spect)
                feature_pool = PixelHop_Neighbour(feature[:, :, :, [c]], dilate, window_size, pad)

                feature_ori[:, :, :, :, c] = feature_pool[:, :, :, :, 0]  # 10, 256, 384, 25

                feature_pool = feature_pool.reshape((-1, feature_pool.shape[-2], feature_pool.shape[-1]))
                n, sp, ch = feature_pool.shape
                print("       <PCA Info>        Pooled feature of PCA:", feature_pool.shape)
                dilate = np.array(dilate).astype('int64')
                all_subTree['leaf' + str(c)] = treeSaab(
                    os.path.join(weight_root + str(hopidx), 'leaf' + str(c) + '.pkl'),
                    kernel_sizes=np.array([2 * window_size + 1]),
                    num_kernels=num_kernels,
                    high_freq_percent=PCA_ener_percent,
                    getcov=getcov,
                    useDC=useDC)
                all_subTree['leaf' + str(c)].fit(feature_pool[:, :, 0])
                total_leaf_num += all_subTree['leaf' + str(c)].leaf_num
                leaf_energy.append(leaf_energy_last[c] * all_subTree[
                    'leaf' + str(c)].energy)  # child energy is multiplied by the parent energy


        else:
            """
            USED IN PIXELHOP 1, NOT CHANNEL-WISE
            """
            feature_pool = PixelHop_Neighbour(feature, dilate, window_size, pad)
            feature_ori = PixelHop_Neighbour(feature, 1, window_size, pad)
            feature_pool = feature_pool.reshape((-1, feature_pool.shape[-2], feature_pool.shape[-1]))  # ***, 25, 3
            n, sp, ch = feature_pool.shape
            print("       <PCA Info>        Pooled feature of PCA:", feature_pool.shape)  # *** ,25,3
            dilate = np.array(dilate).astype('int64')
            all_subTree['leaf0'] = treeSaab(os.path.join(weight_root + str(hopidx), 'leaf0.pkl'),
                                            kernel_sizes=np.array([2 * window_size + 1]),
                                            high_freq_percent=PCA_ener_percent,
                                            getcov=getcov,
                                            useDC=useDC)
            all_subTree['leaf0'].fit(feature_pool.reshape(n, sp * ch))  # ***,75
            total_leaf_num += all_subTree['leaf0'].leaf_num
            leaf_energy = all_subTree['leaf0'].energy
        print("       <PCA Info>        Saab energy in Hop {}".format(hopidx), leaf_energy)

        del feature_pool
        all_subTree['total_leaf_num'] = int(total_leaf_num)
        fwrite = open(os.path.join(weight_root + str(hopidx), 'all_subTree.pkl'), 'wb')
        pickle.dump(all_subTree, fwrite, protocol=2)
        fwrite.close()

        leaf_energy = np.concatenate(leaf_energy, axis=0).squeeze()

        print("Hop " + str(hopidx) + " leaf energy:", leaf_energy)

        fwrite = open(os.path.join(weight_root + str(hopidx), 'leaf_energy.pkl'), 'wb')
        pickle.dump(leaf_energy, fwrite, protocol=2)
        fwrite.close()
        # print("=========== Intermediate and leaf nodes num for Hop: {}".format(total_leaf_num))

    else:
        print(">>>>>====== Testing ======<<<<<")
        # indicator = sample_indicator(gt)
        total_leaf_num = 0.0
        all_subTree = {}
        leaf_energy = []

        if split_spec == 1:
            if hopidx > 1:  # start from 2hop
                # fr = open(weight_root + str(hopidx - 1) + '/leaf_energy.pkl', 'rb')
                # leaf_energy_last = pickle.load(fr)
                # fr.close()
                # not_end_leaf_idx = np.where(leaf_energy_last > Leaf_Ener_thrs)[0].tolist()  # ???
                # print(
                #     "<===================== {} previous leaves has not reach the stopping energy thrs ======================>".format(
                #         len(not_end_leaf_idx)))
                # idx_list = []
                # # ///
                # idx_list = decide_pass_idx1(feature, leaf_energy_last, Leaf_Ener_thrs, Pass_Ener_thrs, hopidx=hopidx)
                # # ///
                fread = open(os.path.join(weight_root + str(hopidx - 1), 'idx_to_pass.pkl'), 'rb')
                idx_list = pickle.load(fread)
                fread.close()
            else:  # 1hop
                idx_list = np.arange(0, feature.shape[-1]).tolist()
            print("<===================== Pass {} previous leaves to this Hop ======================>".format(
                len(idx_list)))
            feature_ori = np.zeros(
                (feature.shape[0], feature.shape[1], feature.shape[2], (2 * window_size + 1) ** 2, feature.shape[-1]))

            for c in idx_list:
                # feature_pool = PixelHop_batchSample(feature[:,:,:,[c]], indicator, dilate, pad) # (n,spacial,spect)
                feature_pool = PixelHop_Neighbour(feature[:, :, :, [c]], dilate, window_size, pad)

                feature_ori[:, :, :, :, c] = feature_pool[:, :, :, :, 0]
        else:
            # feature_pool = PixelHop_batchSample(feature, indicator, dilate, pad) # (n,spacial,spect)
            feature_pool = PixelHop_Neighbour(feature, dilate, window_size, pad)

            feature_ori = feature_pool

    print("=========== End: PixelHop_Unit -> using %10f seconds" % (time.time() - t0))

    return feature_ori, all_subTree


def PixelHop_trans(weight_name, feature_ori, split_spec=1, hopidx=1, Pass_Ener_thrs=0.2, Leaf_Ener_thrs=0.001):
    """
    Transform from the trained saab unit.
    """

    print("------------------- Start: Pixelhop_fit")
    print("       <Fit Info>        Using weight: %s" % str(weight_name))
    t0 = time.time()
    fread = open(os.path.join(weight_name + str(hopidx), 'all_subTree.pkl'), 'rb')
    all_subTree = pickle.load(fread)
    fread.close()
    n, x, y, sp, ch = feature_ori.shape
    # response = np.zeros(feature_ori.shape[0], feature_ori.shape[1], feature_ori.shape[2], all_subTree['total_leaf_num'])
    transformed_feature = []
    transformed_feature_biased = []
    pass_to_next_hop = []
    stay_at_current_hop = []
    if split_spec == 1:
        #        if hopidx>1:
        #            fr = open(weight_name+str(hopidx-1)+'/leaf_energy.pkl', 'rb')
        #            leaf_energy = pickle.load(fr)
        #            fr.close()
        #            idx_list = np.where(leaf_energy>Leaf_Ener_thrs)[0].tolist()
        #        else:
        #            idx_list = np.arange(0,ch).tolist()
        if hopidx > 1:
            fread = open(os.path.join(weight_name + str(hopidx - 1), 'idx_to_pass.pkl'), 'rb')
            idx_list = pickle.load(fread)
            fread.close()
        else:
            idx_list = np.arange(0, ch).tolist()
        for c in idx_list:
            # response[:,:,:,all_subTree['leaf'+str(c)]]
            _, temp_biased = all_subTree['leaf' + str(c)].transform(feature_ori[:, :, :, :, c],
                                                                    Bias=hopidx - 1)  # 1-Hop doesn't use bias
            # temp_biased shape: n, x, y, reduced_sp
            transformed_feature_biased.append(temp_biased)
        transformed_feature_biased = np.concatenate(transformed_feature_biased, axis=-1)
    else:
        transformed_feature, transformed_feature_biased = all_subTree['leaf0'].transform(
            feature_ori.reshape(n, x, y, sp * ch), Bias=hopidx - 1)
        # # show response
        # for sp in range(transformed_feature_biased.shape[-1]):
        #     plt.figure(0)
        #     plt.imshow(transformed_feature_biased[0, :, :, sp], cmap='coolwarm')
        #     plt.colorbar()
        #     plt.savefig(
        #         '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/' + 'DEBUG/response/res_%d_%d.png' % (
        #         0, sp))
        #     plt.close(0)

    print("       <Fit Info>        Transformed feature shape for Hop: {}: %s".format(hopidx) % str(
        transformed_feature_biased.shape))
    print("------------------- End: Pixelhop_fit -> using %10f seconds" % (time.time() - t0))
    return transformed_feature_biased
