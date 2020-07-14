
#%%
import os
import sys
import numpy as np
import time
import pickle
import scipy
import sklearn
import math
import random
import cv2
from skimage import io
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import log_loss as LL
from collections import Counter
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import f1_score as f1
import warnings

warnings.filterwarnings('ignore')





if __name__=='__main__':

    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/multi_resolution/features/'
    print("----Training----")

    # read in multi-resolution features


    # positive1 = np.load(save_dir_matFiles + 'original_resolution/pos_1_100.npy')[:,:,0]
    # negative1 = pickle.load(open(save_dir_matFiles + 'original_resolution/neg_1_100.pkl', 'rb'))[:,:,0]
    positive2 = np.load(save_dir_matFiles + '1-2_resolution/rgb/pos_1_100.npy')[:100000]
    negative2 = pickle.load(open(save_dir_matFiles + '1-2_resolution/rgb/neg_1_100.pkl', 'rb'))[:100000]
    # positive4 = np.load(save_dir_matFiles + 'original_resolution/pos_1_100.npy')[:,:,0]
    # negative4 = pickle.load(open(save_dir_matFiles + 'original_resolution/neg_1_100.pkl', 'rb'))[:,:,0]
    # positive8 = np.load(save_dir_matFiles + '1-8_resolution/pos_1_100.npy')[:,:,0]
    # negative8 = pickle.load(open(save_dir_matFiles + '1-8_resolution/neg_1_100.pkl', 'rb'))[:,:,0]


    # print("original shape:", positive1.shape, negative1.shape)
    print("1/2 shape:", positive2.shape, negative2.shape)
    # print("1/4 shape:", positive4.shape, negative4.shape)
    # print("1/8 shape:", positive8.shape, negative8.shape)


    # X_1 = np.concatenate((positive1, negative1), axis=0)
    X_2 = np.concatenate((positive2, negative2), axis=0)
    # X_4 = np.concatenate((positive4, negative4), axis=0)
    # X_8 = np.concatenate((positive8, negative8), axis=0)



    # target is different with differnt weights
    # pos_target_1 = np.load(save_dir_matFiles + 'original_resolution/pos_target.npy')
    # neg_target_1 = pickle.load(open(save_dir_matFiles + 'original_resolution/neg_target.pkl', 'rb'))
    #
    pos_target_2 = np.load(save_dir_matFiles + '1-2_resolution/rgb/pos_target.npy')[:100000] + 0.25
    neg_target_2 = pickle.load(open(save_dir_matFiles + '1-2_resolution/rgb/neg_target.pkl', 'rb'))[:100000] - 0.25
    #
    # pos_target_4 = np.load(save_dir_matFiles + 'original_resolution/pos_target.npy')
    # neg_target_4 = pickle.load(open(save_dir_matFiles + 'original_resolution/neg_target.pkl', 'rb'))
    #
    # pos_target_8 = np.load(save_dir_matFiles + '1-8_resolution/pos_target.npy')
    # neg_target_8 = pickle.load(open(save_dir_matFiles + '1-8_resolution/neg_target.pkl', 'rb'))


    # n_p, bins_p, patches_p = plt.hist(x=pos_target_2, bins='auto', color='b',
    #                                   alpha=0.7, rwidth=0.85, label="spliced pixels")
    # n_n, bins_n, patches_n = plt.hist(x=neg_target_2, bins='auto', color='y',
    #                                   alpha=0.7, rwidth=0.85, label="authentic pixels")

    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('histogram of train predicted Y for 1/2 resolution')
    # plt.legend()
    #
    # plt.savefig(save_dir_matFiles + "1-2_resolution/rgb/train_target_histogram_1-2.png")


    # read test features, same for different weights
    # test_positive1 = np.load(save_dir_matFiles + 'original_resolution/test_pos_1_40.npy')[:,:,0]
    # test_negative1 = np.load(save_dir_matFiles + 'original_resolution/test_neg_1_40.npy')[:,:,0]
    test_positive2_no = np.load(save_dir_matFiles + '1-2_resolution/rgb/test_pos_1.npy')
    test_negative2_no = np.load(save_dir_matFiles + '1-2_resolution/rgb/test_neg_1.npy')
    test_positive2 = pickle.load(open(save_dir_matFiles + '1-2_resolution/rgb/test_pos_1.pkl', 'rb'))
    test_negative2 = pickle.load(open(save_dir_matFiles + '1-2_resolution/rgb/test_neg_1.pkl', 'rb'))

    # test_positive4 = np.load(save_dir_matFiles + 'original_resolution/test_pos_1_40.npy')[:,:,0]
    # test_negative4 = np.load(save_dir_matFiles + 'original_resolution/test_neg_1_40.npy')[:,:,0]
    # test_positive8 = np.load(save_dir_matFiles + '1-8_resolution/test_pos_1_40.npy')[:,:,0]
    # test_negative8 = np.load(save_dir_matFiles + '1-8_resolution/test_neg_1_40.npy')[:,:,0]

    # print("original shape:", test_positive1.shape, test_negative1.shape)
    print("1/2 shape:", test_positive2[0].shape, test_negative2[0].shape)
    # print("1/4 shape:", test_positive4.shape, test_negative4.shape)
    # print("1/8 shape:", test_positive8.shape, test_negative8.shape)



    # Y_target_1 = np.concatenate((pos_target_1, neg_target_1), axis=0)
    Y_target_2 = np.concatenate((pos_target_2, neg_target_2), axis=0)
    # Y_target_4 = np.concatenate((pos_target_4, neg_target_4), axis=0)
    # Y_target_8 = np.concatenate((pos_target_8, neg_target_8), axis=0)





    #          ###########       TRAINING       ##############

    # split 80% and 20% training and validation set

    # X_train_1, X_val_1, Y_train_1, Y_val_1 = train_test_split(X_1, Y_target_1, test_size=0.1, random_state=42)
    X_train_2, X_val_2, Y_train_2, Y_val_2 = train_test_split(X_2, Y_target_2, test_size=0.1, random_state=42)
    # X_train_4, X_val_4, Y_train_4, Y_val_4 = train_test_split(X_4, Y_target_4, test_size=0.1, random_state=42)
    # X_train_8, X_val_8, Y_train_8, Y_val_8 = train_test_split(X_8, Y_target_8, test_size=0.1, random_state=42)



    # Kmean tree on multi-resolution feats
    num_bin_1 = 64
    num_bin_2 = 64
    num_bin_4 = 16
    num_bin_8 = 8

    # km1 = KMeans(n_clusters = num_bin_1).fit(X_train_1)
    km2 = KMeans(n_clusters = num_bin_2).fit(X_train_2)
    # km4 = KMeans(n_clusters = num_bin_4).fit(X_train_4)
    # km8 = KMeans(n_clusters = num_bin_8).fit(X_train_8)

    # # clusters for km1
    # clus_1 = [[] for i in range(num_bin_1)]
    # regs_1 = [[] for i in range(num_bin_1)]
    # Y_clus_1 = [[] for i in range(num_bin_1)]
    #
    # for k in range(num_bin_1):
    #     clus_samp = X_train_1[km1.labels_ == k]
    #     clus_Y = Y_train_1[km1.labels_ == k]
    #     clus_1[k] = clus_samp
    #     Y_clus_1[k] = clus_Y
    #
    #     reg = RandomForestRegressor(n_estimators=100, n_jobs=8, criterion='mse', min_samples_split=200, min_samples_leaf=100).fit(clus_samp, clus_Y)
    #     regs_1[k] = reg


    # clusters for km2
    clus_2 = [[] for i in range(num_bin_2)]
    Y_clus_2 = [[] for i in range(num_bin_2)]
    regs_2 = [[] for i in range(num_bin_2)]
    for k in range(num_bin_2):
        clus_samp = X_train_2[km2.labels_ == k]
        clus_Y = Y_train_2[km2.labels_ == k]
        clus_2[k] = clus_samp
        Y_clus_2[k] = clus_Y

        reg = LinearRegression().fit(clus_samp, clus_Y)
        regs_2[k] = reg



    # # clusters for km4
    # clus_4 = [[] for i in range(num_bin_4)]
    # Y_clus_4 = [[] for i in range(num_bin_4)]
    # regs_4 = [[] for i in range(num_bin_4)]
    # for k in range(num_bin_4):
    #     clus_samp = X_train_4[km4.labels_ == k]
    #     clus_Y = Y_train_4[km4.labels_ == k]
    #     clus_4[k] = clus_samp
    #     Y_clus_4[k] = clus_Y
    #
    #     reg = RandomForestRegressor(n_estimators=60, n_jobs=8, criterion='mse', min_samples_split=150,
    #                                 min_samples_leaf=75).fit(clus_samp, clus_Y)
    #     regs_4[k] = reg
    #
    # # clusters for km8
    # clus_8 = [[] for i in range(num_bin_8)]
    # Y_clus_8 = [[] for i in range(num_bin_8)]
    # regs_8 = [[] for i in range(num_bin_8)]
    # for k in range(num_bin_8):
    #     clus_samp = X_train_8[km8.labels_ == k]
    #     clus_Y = Y_train_8[km8.labels_ == k]
    #     clus_8[k] = clus_samp
    #     Y_clus_8[k] = clus_Y
    #
    #     reg = RandomForestRegressor(n_estimators=40, n_jobs=8, criterion='mse', min_samples_split=150,
    #                                 min_samples_leaf=75).fit(clus_samp, clus_Y)
    #     regs_8[k] = reg



    # with open(save_dir_matFiles + 'original_resolution/ori_rf.pkl', 'wb') as fid:
    #     pickle.dump(regs_1, fid)

    # with open(save_dir_matFiles + '1-2_resolution/rgb/1-2_LR.pkl', 'wb') as fid:
    #     pickle.dump(regs_2, fid)
    #
    # with open(save_dir_matFiles + '1-4_resolution/1-4_rf.pkl', 'wb') as fid:
    #     pickle.dump(regs_4, fid)
    #
    # with open(save_dir_matFiles + '1-8_resolution/1-8_rf.pkl', 'wb') as fid:
    #     pickle.dump(regs_8, fid)



    Y_train_1_clus_pred = []
    for k in range(num_bin_2):
        Y_train_clus_pre = regs_2[k].predict(clus_2[k])
        Y_train_1_clus_pred.append(Y_train_clus_pre)

    Y_val_1_clus = []
    Y_val_1_clus_pred = []
    Y_val_1_clus_labels = km2.predict(X_val_2)
    for k in range(num_bin_2):
        X_val_clus = X_val_2[Y_val_1_clus_labels == k]
        Y_val_clus = Y_val_2[Y_val_1_clus_labels == k]
        Y_val_1_clus.append(Y_val_clus)

        Y_val_clus_pre = regs_2[k].predict(X_val_clus)
        Y_val_1_clus_pred.append(Y_val_clus_pre)

    Y_clus_1 = np.concatenate(Y_clus_2, axis=0)    # gt train
    Y_train_1_clus_pred = np.concatenate(Y_train_1_clus_pred, axis=0)    # train pred

    Y_val_1_clus = np.concatenate(Y_val_1_clus, axis=0)  # gt val
    Y_val_1_clus_pred = np.concatenate(Y_val_1_clus_pred, axis=0)   # val pred








    #%%


    Y_val_label = np.zeros((Y_val_1_clus.shape))
    Y_train_label = np.zeros((Y_clus_1.shape))

    # gt
    Y_val_label[Y_val_1_clus > 0] = 1
    Y_train_label[Y_clus_1 > 0] = 1


    # pred
    Y_val_predicted_label = np.zeros((Y_val_1_clus_pred.shape))
    Y_train_predicted_label = np.zeros((Y_train_1_clus_pred.shape))

    Y_val_predicted_label[Y_val_1_clus_pred > 0] = 1
    Y_train_predicted_label[Y_train_1_clus_pred > 0] = 1

    # plt.figure()
    # #   histogram of predicted train
    # n_p, bins_p, patches_p = plt.hist(x=Y_train_1_clus_pred, bins='auto', color='b',
    #                                   alpha=0.7, rwidth=0.85, label="spliced pixels")
    # # n_n, bins_n, patches_n = plt.hist(x=neg_pred, bins='auto', color='y',
    # #                                   alpha=0.7, rwidth=0.85, label="authentic pixels")
    #
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('histogram of train predicted Y for 1/2 resolution')
    # plt.legend()
    #
    # plt.savefig(save_dir_matFiles + "1-2_resolution/rgb/train_histogram_1-2.png")



    C_train = metrics.confusion_matrix(Y_train_label, Y_train_predicted_label, labels=[0, 1])
    per_class_accuracy_train = np.diag(C_train.astype(np.float32)) / np.sum(C_train.astype(np.float32), axis=1)
    print("train:", per_class_accuracy_train)

    C_val = metrics.confusion_matrix(Y_val_label, Y_val_predicted_label, labels=[0, 1])
    per_class_accuracy_validation = np.diag(C_val.astype(np.float32)) / np.sum(C_val.astype(np.float32), axis=1)
    print("validation:", per_class_accuracy_validation)









            ############        TESTING        ##############
    print("----TEST----")
    name_loc_prob = pickle.load(open(save_dir_matFiles + '1-2_resolution/rgb/name_loc.pkl', 'rb'))

    counts = np.zeros((len(name_loc_prob), 2))
    for k in range(len(name_loc_prob)):
        counts[k][0] = len(name_loc_prob[k]['spliced_loc'])
        counts[k][1] = len(name_loc_prob[k]['authentic_loc'])

    print("total number of spliced and authentic pixels in testing images:", np.sum(counts, axis=0))


    counts_splice = counts[:, 0]
    counts_authen = counts[:, 1]




    # Y_test_all_image = []
    # Y_test_predict_all_image = []
    test_pos_pred_all_image = []
    test_neg_pred_all_image = []

    test_pos_gt_all_image = []
    test_neg_gt_all_image = []


    for k in range(len(name_loc_prob)):
        print("-- processing test image {} :".format(k))
        test_pos_loc = name_loc_prob[k]['spliced_loc']
        test_neg_loc = name_loc_prob[k]['authentic_loc']
        # X_test_loc = np.concatenate((test_pos_loc, test_neg_loc), axis=0)
        test_pos_loc = np.array(test_pos_loc)
        test_neg_loc = np.array(test_neg_loc)
        print(test_pos_loc.shape, test_neg_loc.shape)

        pos_num = len(test_pos_loc)
        neg_num = len(test_neg_loc)

        # test_positive_1_k = test_positive2[int(np.sum(counts_splice[:k])): int(np.sum(counts_splice[:k])) + pos_num, :]
        # test_negative_1_k = test_negative2[int(np.sum(counts_authen[:k])): int(np.sum(counts_authen[:k])) + neg_num, :]
        test_positive_1_k = test_positive2[k]
        test_negative_1_k = test_negative2[k]
        print(test_positive_1_k.shape, test_negative_1_k.shape)

        test_pos_labels = np.ones((test_positive_1_k.shape[0]))
        test_neg_labels = np.zeros((test_negative_1_k.shape[0]))
        # Y_test_k = np.concatenate((test_pos_labels, test_neg_labels), axis=0)

        # X_test_k = np.concatenate((test_positive_1_k, test_negative_1_k), axis=0)   # training sample for each image


        ##### positive #####
        Y_test_clus_pre = []  # every image
        Y_test_clus = []  # every image
        loc_test_clus = [] # every image


        test_pos_k_clus_labels = km2.predict(test_positive_1_k)

        for i in range(num_bin_2):

            X_test_clus = test_positive_1_k[test_pos_k_clus_labels == i]
            if X_test_clus.shape[0] != 0:
                Y_test_clus.append(test_pos_labels[test_pos_k_clus_labels == i])  # binary
                loc_test_clus.append(test_pos_loc[test_pos_k_clus_labels == i])

                tmp = regs_2[i].predict(X_test_clus)  # continuous
                Y_test_clus_pre.append(tmp)

        Y_test_clus = np.concatenate(Y_test_clus)  # binary
        Y_test_clus_pre = np.concatenate(Y_test_clus_pre) # continuous
        test_pos_pred_all_image.extend(Y_test_clus_pre)
        test_pos_gt_all_image.extend(Y_test_clus)


        # write in name_loc_prob    X_test_loc  & Y_test_clus_pre
        for l in range(Y_test_clus_pre.shape[0]):     #
            loc = test_pos_loc[l].tolist()     # location to corresponding

            # if loc in name_loc_prob[k]['spliced_loc']:
            idx = name_loc_prob[k]['spliced_loc'].index(loc)

            name_loc_prob[k]['spliced_loc'][idx].append(Y_test_clus_pre[l])






        # print("##### negetive #####")
        Y_test_clus_pre = []  # every image
        Y_test_clus = []  # every image
        loc_test_clus = [] # every image


        test_neg_k_clus_labels = km2.predict(test_negative_1_k)

        for i in range(num_bin_2):

            X_test_clus = test_negative_1_k[test_neg_k_clus_labels == i]
            if X_test_clus.shape[0] != 0:
                Y_test_clus.append(test_neg_labels[test_neg_k_clus_labels == i])  # binary
                loc_test_clus.append(test_neg_loc[test_neg_k_clus_labels == i])

                tmp = regs_2[i].predict(X_test_clus)  # continuous
                Y_test_clus_pre.append(tmp)

        Y_test_clus = np.concatenate(Y_test_clus)  # binary
        Y_test_clus_pre = np.concatenate(Y_test_clus_pre) # continuous
        test_neg_pred_all_image.extend(Y_test_clus_pre)
        test_neg_gt_all_image.extend(Y_test_clus)


        # write in name_loc_prob    X_test_loc  & Y_test_clus_pre
        for l in range(Y_test_clus_pre.shape[0]):     #
            loc = test_neg_loc[l].tolist()     # location to corresponding

            # if loc in name_loc_prob[k]['authentic_loc']:

            idx = name_loc_prob[k]['authentic_loc'].index(loc)

            name_loc_prob[k]['authentic_loc'][idx].append(Y_test_clus_pre[l])




    Y_test_pred_label = np.zeros((test_positive2_no.shape[0] + test_negative2_no.shape[0]))
    pred = np.concatenate((test_pos_pred_all_image, test_neg_pred_all_image), axis=0)
    Y_test_pred_label[pred > 0] = 1

    gt = np.concatenate((test_pos_gt_all_image, test_neg_gt_all_image), axis=0)










        # # write in name_loc_prob    X_test_loc  & Y_test_clus_pre
        # for l in range(X_test_loc.shape[0]):
        #     loc = X_test_loc[l].tolist()
        #
        #     if loc in name_loc_prob[k]['spliced_loc']:
        #         idx = name_loc_prob[k]['spliced_loc'].index(loc)
        #         name_loc_prob[k]['spliced_loc'][idx].append(Y_test_clus_pre[l])
        #
        #     if loc in name_loc_prob[k]['authentic_loc']:
        #         idx = name_loc_prob[k]['authentic_loc'].index(loc)
        #         name_loc_prob[k]['authentic_loc'][idx].append(Y_test_clus_pre[l])



    C_test = metrics.confusion_matrix(gt, Y_test_pred_label, labels=[0, 1])
    per_class_accuracy_test = np.diag(C_test.astype(np.float32)) / np.sum(C_test.astype(np.float32), axis=1)
    print("test acc before double threshold:", per_class_accuracy_test)



    test_pos_pred_all_image = np.array(test_pos_pred_all_image)
    test_neg_pred_all_image = np.array(test_neg_pred_all_image)
    print(np.max(test_pos_pred_all_image), np.min(test_pos_pred_all_image))
    print(np.max(test_neg_pred_all_image), np.min(test_neg_pred_all_image))



    plt.figure()
    n_p, bins_p, patches_p = plt.hist(x=test_pos_pred_all_image, bins='auto', color='b',
                                    rwidth=0.85, label="spliced pixels")
    n_n, bins_n, patches_n = plt.hist(x=test_neg_pred_all_image, bins='auto', color='y',
                                    rwidth=0.85, label="authentic pixels")

    plt.grid(axis='y', alpha=0.75)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('histogram of testing predicted Y for 1/2 resolution')
    plt.legend()

    plt.savefig(save_dir_matFiles + "1-2_resolution/rgb/histogram_1-2.png")




    #
    # with open(save_dir_matFiles + '1-2_resolution/rgb/name_loc_prob.pkl', 'wb') as fid:
    #     pickle.dump(name_loc_prob, fid)

#%%
    # with open(save_dir_matFiles + '1-2_resolution/rgb/name_loc_prob.pkl', 'rb') as fid:
    #     name_loc_prob = pickle.load(fid)



    output_prob_map = np.zeros((len(name_loc_prob), 128, 192))
    gt_map = np.zeros((len(name_loc_prob), 128, 192))

    for k in range(len(name_loc_prob)):
        # name_loc_prob[k] is a dictionary
        splice_pixel_loc = name_loc_prob[k]['spliced_loc']    # list
        authen_pixel_loc = name_loc_prob[k]['authentic_loc']    # list
        for pos_pixel in range(len(splice_pixel_loc)):
            i = splice_pixel_loc[pos_pixel][0]
            j = splice_pixel_loc[pos_pixel][1]
            output_prob_map[k, i, j] = splice_pixel_loc[pos_pixel][-1]
            gt_map[k,i,j] = 1

        for neg_pixel in range(len(authen_pixel_loc)):
            i = authen_pixel_loc[neg_pixel][0]
            j = authen_pixel_loc[neg_pixel][1]
            output_prob_map[k, i, j] = authen_pixel_loc[neg_pixel][-1]
            gt_map[k,i,j] = -1

    for k in range(len(name_loc_prob)):
        plt.figure(0)
        plt.imshow(output_prob_map[k], cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.savefig(save_dir_matFiles + '1-2_resolution/rgb/visual/' + name_loc_prob[k]['test_name'][:-4] + '_output_probmap_1-2_reso.png')
        plt.close(0)

        # plt.figure(0)
        # plt.imshow(gt_map[k], cmap='coolwarm')
        # plt.colorbar()
        # plt.savefig(save_dir_matFiles + '1-2_resolution/rgb/visual/' + name_loc_prob[k]['test_name'][:-4] + '_1-2_gt_map.png')
        # plt.close(0)



#%%

    # ensemble result from original resolution

    # read in images of s1a1 test result
    name_loc_prob_ori = pickle.load(open('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/nonoverlap_s1a1/columbia/name_loc_prob.pkl', 'rb'))
    output_prob_map_ori = np.zeros((len(name_loc_prob_ori), 256, 384))
    gt_map_ori = np.zeros((len(name_loc_prob_ori), 256, 384))

    for k in range(len(name_loc_prob_ori)):
        # name_loc_prob[k] is a dictionary
        splice_pixel_loc = name_loc_prob_ori[k]['spliced_loc']    # list
        authen_pixel_loc = name_loc_prob_ori[k]['authentic_loc']    # list
        for pos_pixel in range(len(splice_pixel_loc)):
            i = splice_pixel_loc[pos_pixel][0]
            j = splice_pixel_loc[pos_pixel][1]
            output_prob_map_ori[k, i, j] = splice_pixel_loc[pos_pixel][2]
            gt_map_ori[k,i,j] = 1

        for neg_pixel in range(len(authen_pixel_loc)):
            i = authen_pixel_loc[neg_pixel][0]
            j = authen_pixel_loc[neg_pixel][1]
            output_prob_map_ori[k, i, j] = authen_pixel_loc[neg_pixel][2]
            gt_map_ori[k,i,j] = -1

    # interpolate low resolution result
    resized = np.zeros((output_prob_map.shape[0], 256, 384))
    for k in range(output_prob_map.shape[0]):
        resized[k] = cv2.resize(output_prob_map[k], (384, 256))


    # ensemble result: simply compare
    ensemble = np.zeros((output_prob_map_ori.shape))
    for k in range(len(output_prob_map)):
        ensemble[k] = np.maximum(output_prob_map_ori[k], resized[k])





    for k in range(len(name_loc_prob)):
        plt.figure(0)
        plt.imshow(ensemble[k], cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.savefig(save_dir_matFiles + 'models/ensemble/' + name_loc_prob[k]['test_name'][:-4] + '_output_probmap_hop2_ensemble.png')
        plt.close(0)







