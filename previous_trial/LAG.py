#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:49:47 2019

@author: yueru
"""

import scipy
from sklearn import preprocessing
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.cluster import MiniBatchKMeans, KMeans


def Norm(feature):
    #   shape:(# sample, # feature)
    feature = feature - np.min(feature, 1).reshape(-1, 1)
    feature = feature / np.sum(feature, 1).reshape(-1, 1)
    return feature


def Relu(centroid_old):
    #   shape:(# sample, # feature)
    centroid_old[centroid_old < 0] = 0
    return centroid_old


def llsr_train(feature_train, labels_train, encode=True, centroid=None, clus_labels=None, train_labels=None,
               scaler=None, alpha=10):
    # llsr_train(feature,labels_train.astype(int),encode=True,centroid=centroid, clus_labels=clus_labels,train_labels=train_labels, scaler=scaler,alpha=alpha)

    if encode:
        alpha = alpha
        print('Alpha:', alpha)
        #        labels_train_onehot = encode_onehot(labels_train)
        n_sample = labels_train.shape[0]  # 1000
        labels_train_onehot = np.zeros((n_sample, clus_labels.shape[0]))  # 1000, 50
        for i in range(n_sample):  # 1000
            gt = train_labels[i]  # sample i's class label
            idx = clus_labels == gt
            dis = euclidean_distances(feature_train[i].reshape(1, -1),
                                      centroid[idx])  # distance from one sample to 5 cluster centroid (1,5)
            dis = dis.reshape(-1)  # (5)
            dis = dis / (dis.min() + 1e-5)
            p_dis = np.exp(-dis * alpha)
            p_dis = p_dis / p_dis.sum()
            labels_train_onehot[i, idx] = p_dis

    else:
        labels_train_onehot = labels_train
    feature_train = scaler.transform(feature_train)  # standarize features
    A = np.ones((feature_train.shape[0], 1))  # 1000, 1  bias term
    feature_train = np.concatenate((A, feature_train), axis=1)
    #    print(np.sort(labels_train_onehot[:10],1)[:,::-1])
    weight = scipy.linalg.lstsq(feature_train, labels_train_onehot)[0]
    weight_save = weight[1:weight.shape[0]]
    bias_save = weight[0].reshape(1, -1)
    return weight_save, bias_save


def llsr_test(feature_test, weight_save, bias_save):
    feature_test = np.matmul(feature_test, weight_save)
    feature_test = feature_test + bias_save
    return feature_test


def compute_target_(feature, train_labels, num_clusters, class_list):
    use_classes = len(class_list)  # 10

    train_labels = train_labels.reshape(-1)
    # num_clusters_sub = int(num_clusters/use_classes)  # 50/10 = 5
    total_cluster_num = sum(num_clusters)
    batch_size = 1000
    labels = np.zeros((feature.shape[0]))  # 1000
    clus_labels = np.zeros((total_cluster_num,))  # 50
    centroid = np.zeros((total_cluster_num, feature.shape[1]))  # 50, 320

    for i in range(use_classes):
        ID = class_list[i]
        feature_train = feature[train_labels == ID]
        kmeans = MiniBatchKMeans(n_clusters=num_clusters[i], batch_size=batch_size).fit(feature_train)
        #            kmeans = KMeans(n_clusters=num_clusters_sub).fit(feature_train)
        labels[train_labels == ID] = kmeans.labels_ + sum(num_clusters[:i])  # i*num_clusters_sub
        clus_labels[sum(num_clusters[:i]):sum(num_clusters[:(i + 1)])] = ID
        centroid[sum(num_clusters[:i]):sum(num_clusters[:(i + 1)])] = kmeans.cluster_centers_
        print('FINISH KMEANS', i, '   cluster num', num_clusters[i])

    return labels, clus_labels.astype(int), centroid


def encode_onehot(a):
    a = a.reshape(-1)
    print('before encode shape:', a.shape)
    b = np.zeros((a.shape[0], 1 + int(a.max())))  # - 1./a.max()
    b[np.arange(a.shape[0]), a] = 1
    print('after encode shape:', b.shape)
    return b.astype(float)


def calculate_cross_entropy(data, label, class_frequency, bins=10):
    # Function to calculate cross entropy of feature dimensions

    # Input:
    # data - output of LAG (nunber of training samples, number of dimensions)
    # label - class labels (nunber of training samples, 1)
    # bins - Number of bins for histogram (default - 10 bins)
    # class_frequency - array containing number of samples for each class (number of classes, 1)

    # Output:
    # cross_entropy - normalized cross_entropy of every dimension in the data (number of dimesnsions, )

    n_classes = np.unique(label).shape[0]
    mini = data.min(axis=0).reshape((1, data.shape[1]))
    maxi = data.max(axis=0).reshape((1, data.shape[1]))
    data = (data - mini) / (maxi - mini)

    binwise_frequency = np.zeros((bins, data.shape[1], n_classes))  # 10, 30, 2

    for i in range(data.shape[1]):  # number of features
        for j in range(data.shape[0]):  # number of samples
            desired_bin = np.ceil(data[j][i] * bins) - 1
            binwise_frequency[int(desired_bin)][i][label[j]] += 1

    binwise_frequency = binwise_frequency / class_frequency
    binwise_class = np.argmax(binwise_frequency, axis=2)
    print("bin wise class shape:", binwise_class.shape)

    correct_class_probability = np.zeros((data.shape[1], data.shape[0]))  # p(n,c) 30, 20000
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            correct_class_probability[i][j] = np.count_nonzero(binwise_class.T[i] == label[j]) / bins

    log = -1 * np.log(correct_class_probability + 1e-9)
    cross_entropy = np.sum(log, axis=1)
    # mini = np.amin(cross_entropy)
    # maxi = np.amax(cross_entropy)
    # cross_entropy = (cross_entropy - mini) / (maxi - mini)

    return cross_entropy


def LAG_Unit(feature, train_labels=None, class_list=None, class_frequency=None, SAVE=None, num_clusters=None, alpha=5, Train=True):
    # train_feature_reduce=LAG_Unit(train_feature,train_labels=train_labels, class_list=class_list,
    #                              SAVE=SAVE,num_clusters=50,alpha=5,Train=True)

    #                  feature: training features or testing features    (1000, 320)
    #                  train_labels: real labels of samples
    #                  class_list: list of class labels   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #                  SAVE: store parameters   {}
    #                  num_clusters: output feature dimension   # <-- [5,3,3,3,3,3,3,3,3,3]
    #                  alpha: a parameter when computing probability vector
    #                  Train: True: training stage; False: testing stage

    if Train:

        print('--------Train LAG Unit--------')
        print('feature_train shape:', feature.shape)  # 1000, 320
        use_classes = len(np.unique(train_labels))  # 10
        k = 0
        # Compute output features
        labels_train, clus_labels, centroid = compute_target_(feature, train_labels, num_clusters,
                                                              class_list)
        unique, counts = np.unique(labels_train, return_counts=True)
        idx_samp_in_cluster = counts < 50
        print("labels_train count:", dict(zip(unique, counts)))
        print("clus_labels shape:", clus_labels.shape)
        print("centroid shape:", centroid.shape)
        #                SAVE['train_dis']=cosine_similarity(feature_train,centroid)
        #                SAVE['test_dis']=cosine_similarity(feature_test,centroid)
        # Solve LSR
        scaler = preprocessing.StandardScaler().fit(feature)
        # scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        weight_save, bias_save = llsr_train(feature, labels_train.astype(int), encode=True, centroid=centroid,
                                            clus_labels=clus_labels, train_labels=train_labels,
                                            scaler=scaler, alpha=alpha)

        SAVE[str(k) + ' clus_labels'] = clus_labels
        SAVE[str(k) + ' LLSR weight'] = weight_save
        SAVE[str(k) + ' LLSR bias'] = bias_save
        SAVE[str(k) + ' scaler'] = scaler

        feature_reduce = llsr_test(scaler.transform(feature), weight_save, bias_save)
        cross_entropy = calculate_cross_entropy(feature_reduce, train_labels, class_frequency)
        print("cross entropy:", cross_entropy)

        # pred_labels = np.zeros((feature.shape[0], use_classes))
        # for km_i in range(use_classes):
        #     pred_labels[:, km_i] = feature[:, clus_labels == km_i].sum(1)
        # pred_labels = np.argmax(pred_labels, axis=1)
        # idx = pred_labels == train_labels.reshape(-1)
        # print(k, ' Kmean training acc is: {}'.format(1. * np.count_nonzero(idx) / train_labels.shape[0]))

        #iteration
        threshold = 200000
        print("threshold:", threshold)
        train_feature_reduce = []
        for xx in range(1,11):

            print("----iteration xx:", xx)

            # idx = cross_entropy > threshold
            idx_tf = np.where(cross_entropy == np.amax(cross_entropy))
            idx = [False for i in range(cross_entropy.shape[0])]
            for i in range(idx_tf[0].shape[0]):
                idx[idx_tf[0][i]] = True


            if np.sum((cross_entropy > threshold) == True) == 0 or np.sum(idx_samp_in_cluster == True) > 0:
                print("all cross entropy values below threshold", xx)
                break

            new_clus_labels = np.zeros((clus_labels.shape[0] + np.sum(idx)))   # 4 -> 7
            new_labels_train = np.zeros((feature.shape[0]))     # 1000
            new_centroid = np.zeros((clus_labels.shape[0] + np.sum(idx), feature.shape[1]))

            for i in range(cross_entropy.shape[0]):
                loc = np.sum(idx[:i]) * 2 + (i-np.sum(idx[:i]))
                print(loc)
                loc = int(loc)
                if idx[i] == True:
                    feat_to_split = feature[labels_train == i]
                    print("----", feat_to_split.shape)
                    kmeans = KMeans(n_clusters=2, n_init=20).fit(feat_to_split)

                    new_labels_train[labels_train == i] = kmeans.labels_ + loc
                    new_clus_labels[loc : loc + 2] = clus_labels[i]
                    new_centroid[loc : loc + 2] = kmeans.cluster_centers_

                else:
                    new_clus_labels[loc] = clus_labels[i]
                    new_labels_train[labels_train == i] = loc
                    new_centroid[loc] = centroid[i]

            print("----new cluster labels:", new_clus_labels)
            print("----new number of clusters:", new_centroid.shape[0])

            labels_train = new_labels_train
            unique, counts = np.unique(labels_train, return_counts=True)
            print("----labels_train count:", dict(zip(unique, counts)))
            idx_samp_in_cluster = counts < 50
            clus_labels = new_clus_labels
            centroid = new_centroid

            weight_save, bias_save = llsr_train(feature, labels_train.astype(int), encode=True, centroid=centroid,
                                                clus_labels=clus_labels, train_labels=train_labels,
                                                scaler=scaler, alpha=alpha)

            SAVE[str(xx) + ' clus_labels'] = clus_labels
            SAVE[str(xx) + ' LLSR weight'] = weight_save
            SAVE[str(xx) + ' LLSR bias'] = bias_save
            SAVE[str(xx) + ' scaler'] = scaler

            feature_reduce = llsr_test(scaler.transform(feature), weight_save, bias_save)
            cross_entropy = calculate_cross_entropy(feature_reduce, train_labels, class_frequency)
            print("----cross entropy:", cross_entropy)

            train_feature_reduce.append(feature_reduce)


        return train_feature_reduce

    else:
        print('--------Testing--------')
        # iteration
        test_feature_reduce = []
        print("SAVE dict length:", int(len(SAVE)/4))
        for k in range(1,int(len(SAVE)/4)):
            scaler = SAVE[str(k) + ' scaler']
            feature_reduced = llsr_test(scaler.transform(feature), SAVE[str(k) + ' LLSR weight'],
                                        SAVE[str(k) + ' LLSR bias'])
            test_feature_reduce.append(feature_reduced)

        #
        # k = 0
        # scaler = SAVE[str(k) + ' scaler']
        # feature_reduced = llsr_test(scaler.transform(feature), SAVE[str(k) + ' LLSR weight'],
        #                             SAVE[str(k) + ' LLSR bias'])
        return test_feature_reduce


