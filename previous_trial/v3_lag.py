#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:19:51 2019

@author: mac
"""

import numpy as np
import _pickle as cPickle
from skimage import io
from PCA import saab
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import multiprocessing
from sklearn.metrics import confusion_matrix
import math
import cv2
from sklearn.cluster import KMeans
import scipy

#%%

def RobertsOperator(roi):
    operator_first = np.array([[-1, 0], [0, 1]])
    operator_second = np.array([[0, -1], [1, 0]])
    return np.abs(np.sum(roi[1:, 1:] * operator_first)) + np.abs(np.sum(roi[1:, 1:] * operator_second))

def RobertsAlogrithm(image):
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            image[i, j] = RobertsOperator(image[i - 1:i + 2, j - 1:j + 2])
    return image[1:image.shape[0] - 1, 1:image.shape[1] - 1]

def boundary_extension(image_dir, filenames):
    # image_dir = train_dir+img_dir
    
    n_samp = len(filenames)
    filenames_filtered = [filenames[i] for i in range(n_samp)]
    extended_imgs = []
    for i in range(n_samp):
        img_samp = io.imread(image_dir + filenames_filtered[i])
#        print(img_samp.shape)
        padded_img_samp = np.lib.pad(img_samp, ((4,4),(4,4),(0,0)), 'symmetric') # symmetric padding 
#        print(padded_img_samp.shape)
        extended_imgs.append(padded_img_samp)
        
    extended_imgs = np.array(extended_imgs)
    
    return extended_imgs

def read_gc(edges_dir, filenames):
    
    n_samp = len(filenames)
    filenames_filtered = [filenames[i] for i in range(n_samp)]
    gc = []
    for i in range(n_samp):
        edge_samp = io.imread(edges_dir + filenames_filtered[i][:-4]+'_gc.png')
        gc.append(edge_samp)
    gc = np.array(gc)
    
    return gc

def read_all_edge(edges_dir, filenames):
    
    n_samp = len(filenames)
    filenames_filtered = [filenames[i] for i in range(n_samp)]
    alledge = []
    for i in range(n_samp):
        edge_samp = io.imread(edges_dir + filenames_filtered[i])
        alledge.append(edge_samp)
    alledge = np.array(alledge)
    
    return alledge
        

def Saab_Getkernel(data, num_kernels):
    # read data
    # reshape data
    train_images = data.reshape((data.shape[0]*data.shape[1]*data.shape[2], int(math.sqrt(data.shape[3]/3)), int(math.sqrt(data.shape[3]/3)), 3))
    #shape: 100*128*192, 3,3,3
    train_labels =np.ones((train_images.shape[0],))
    kernel_sizes = str(train_images.shape[1])
    
    
#    if flag == 'pos':
#        train_labels = np.ones((train_images.shape[0],), dtype=int)
#        
#    else:
#        train_labels = np.zeros((train_images.shape[0],), dtype=int)
#    test_labels = np.load('new_data_result/test_label.npy')
    class_list = [0,1]

    print('       START SAAB TRAINING KERNEL       ')
    print('       Training image size:', train_images.shape)
#    print('Testing_image size:', test_images.shape)
    kernel_sizes= saab.parse_list_string(kernel_sizes)
    if num_kernels:
        num_kernels= saab.parse_list_string(num_kernels)
    else:
        num_kernels=None
    energy_percent=None
    use_num_images=-1
    print('       Parameters:')
    print('       use_classes:', class_list)
    print('       Kernel_sizes:', kernel_sizes)
    print('       Number_kernels:', num_kernels)
    print('       Energy_percent:', energy_percent)
    print('       Number_use_images:', use_num_images)

    pca_params= saab.multi_Saab_transform(train_images, train_labels,
                                          kernel_sizes=kernel_sizes,
                                          num_kernels=num_kernels,
                                          energy_percent=energy_percent,
                                          use_num_images=use_num_images,
                                          use_classes=class_list)
#    # save data
#    fw=open('Saab_output/' + neg_filter_test10.pkl','wb')
#    pickle.dump(pca_params, fw)    
#    fw.close()
    return pca_params


def Saab_Getfeature(data, filters):
    
    train_images = data.reshape((data.shape[0]*data.shape[1]*data.shape[2], int(math.sqrt(data.shape[3]/3)), int(math.sqrt(data.shape[3]/3)), 3))
    #shape: 100*128*192, 3,3,3
    
    pca_params = filters

    print('       Training image size:', train_images.shape)
#    print('Testing_image size:', test_images.shape)

    
    # Training
    print('       Training feature extraction--------')
    feature = saab.initialize(train_images, pca_params) 
    print("       S4 shape:", feature.shape)
    print('       Finish Feature Extraction subnet--------')
    
    feature_out = feature.reshape((data.shape[0], data.shape[1], data.shape[2], feature.shape[-1]))
    
    return feature_out

def one_hop(extended_imgs):
    
    # 8 nearest neighbors
    n_samp = extended_imgs.shape[0]
    h = extended_imgs.shape[1] #264
    w = extended_imgs.shape[2] #392
    
    one_hop_attri_saab = np.zeros((n_samp, h-8,w-8, 27))
    
    for k in range(n_samp):
        for i in range(4, h-4):
            for j in range(4,w-4):
                top = np.concatenate((extended_imgs[k, i-1, j-1, :], extended_imgs[k,i-1,j,:], extended_imgs[k,i-1,j+1,:]), axis=None)
                
                middle = np.concatenate((extended_imgs[k, i, j-1, :], extended_imgs[k,i,j,:], extended_imgs[k,i,j+1,:]), axis=None)
                
                bottom = np.concatenate((extended_imgs[k, i+1, j-1, :], extended_imgs[k,i+1,j,:], extended_imgs[k,i+1,j+1,:]), axis=None)
                one_hop_attri_saab[k,i-4,j-4,:] = np.concatenate((top, middle, bottom), axis=None)
#                print((np.concatenate((top, middle, bottom), axis=None)).shape)  # 27
    
    # Pooling from 100,256,384,27 to 100,128,192,27
    # block_reduce(one_hop_attri_saab, block_size=(4,4,1), func=np.mean)
    pooled_attri = np.zeros((n_samp, int(one_hop_attri_saab.shape[1]/2), int(one_hop_attri_saab.shape[2]/2), one_hop_attri_saab.shape[3]))
    for i in range(0, one_hop_attri_saab.shape[1], 2):
        for j in range(0, one_hop_attri_saab.shape[2], 2):
            pooled_attri[:,int(i/2),int(j/2),:] = one_hop_attri_saab[:,i,j,:]
    print("pooled attribute 1 shape:", pooled_attri.shape)
                
    # Saab dimension reduction from 27 dim to 12 dim
    pca_params_1 = Saab_Getkernel(pooled_attri, "11")
    one_hop_attri = Saab_Getfeature(pooled_attri, pca_params_1)
    
    
    print("one hop attribute shape:", one_hop_attri.shape)
  
    
    return pca_params_1, one_hop_attri

def second_hop(one_hop_attri):
    
    # 8 nearest neighbors
    n_samp = one_hop_attri.shape[0]
    h = one_hop_attri.shape[1] #128
    w = one_hop_attri.shape[2] #192
    
    #one_hop_attri shape: 100, 256, 384, 12
    padded_one_hop_attri = np.lib.pad(one_hop_attri, ((0,0), (2,2), (2,2), (0,0)), 'symmetric')   
    print("padded attribute shape", padded_one_hop_attri.shape) # 100, 260, 388, 12
    
    sec_hop_attri_saab = np.zeros((n_samp, h, w, 108))
    
    for k in range(n_samp):
        for i in range(2,2+h): #2-258
            for j in range(2,2+w): #2-386
                atttotal = []
                for m in range(12):
                    top = np.concatenate((padded_one_hop_attri[k,i-2,j-2,m], padded_one_hop_attri[k,i-2,j,m], padded_one_hop_attri[k,i-2,j+2,m]), axis=None)
                    mid = np.concatenate((padded_one_hop_attri[k,i,j-2,m], padded_one_hop_attri[k,i,j,m], padded_one_hop_attri[k,i,j+2,m]), axis=None)
                    bot = np.concatenate((padded_one_hop_attri[k,i+2,j-2,m], padded_one_hop_attri[k,i+2,j,m], padded_one_hop_attri[k,i+2,j+2,m]), axis=None)
                    attslice = np.concatenate((top, mid, bot), axis=None) #9
                    atttotal.extend(attslice)
                atttotal = np.array(atttotal)
                sec_hop_attri_saab[k,i-2,j-2,:] = atttotal
                
    pooled_attri_2 = np.zeros((n_samp, int(sec_hop_attri_saab.shape[1]/2), int(sec_hop_attri_saab.shape[2]/2), sec_hop_attri_saab.shape[3]))
    for i in range(0, sec_hop_attri_saab.shape[1], 2):
        for j in range(0, sec_hop_attri_saab.shape[2], 2):
            pooled_attri_2[:,int(i/2),int(j/2),:] = sec_hop_attri_saab[:,i,j,:]
    print("pooled attribute 2 shape:", pooled_attri_2.shape)
                
    # Saab dimension reduction from 27 dim to 12 dim
    pca_params_2 = Saab_Getkernel(pooled_attri_2, "47")
    sec_hop_attri = Saab_Getfeature(pooled_attri_2, pca_params_2)
                    
    print("second hop attribute shape:", sec_hop_attri.shape)
    
    return pca_params_2, sec_hop_attri


def third_hop(sec_hop_attri):
    
    # 8 nearest neighbors
    n_samp = sec_hop_attri.shape[0]
    h = sec_hop_attri.shape[1] #64
    w = sec_hop_attri.shape[2] #96
    
    #sec_hop_attri shape: 100, 64, 96, 50
    padded_sec_hop_attri = np.lib.pad(sec_hop_attri, ((0,0), (2,2), (2,2), (0,0)), 'symmetric')   
    print("padded attribute shape", padded_sec_hop_attri.shape) # 100, 264, 392, 50
    
    third_hop_attri_saab = np.zeros((n_samp, h, w, sec_hop_attri.shape[-1]*9))
    
    for k in range(n_samp):
        for i in range(2,2+h): #4-260
            for j in range(2,2+w): #4-388
                atttotal = []
                for m in range(sec_hop_attri.shape[-1]):
                    top = np.concatenate((padded_sec_hop_attri[k,i-2,j-2,m], padded_sec_hop_attri[k,i-2,j,m], padded_sec_hop_attri[k,i-2,j+2,m]), axis=None)
                    mid = np.concatenate((padded_sec_hop_attri[k,i,j-2,m], padded_sec_hop_attri[k,i,j,m], padded_sec_hop_attri[k,i,j+2,m]), axis=None)
                    bot = np.concatenate((padded_sec_hop_attri[k,i+2,j-2,m], padded_sec_hop_attri[k,i+2,j,m], padded_sec_hop_attri[k,i+2,j+2,m]), axis=None)
                    attslice = np.concatenate((top, mid, bot), axis=None) #9
                    atttotal.extend(attslice)
                atttotal = np.array(atttotal)
                third_hop_attri_saab[k,i-2,j-2,:] = atttotal
    
#    pooled_attri_3 = np.zeros((n_samp, int(third_hop_attri_saab.shape[1]/2), int(third_hop_attri_saab.shape[2]/2), third_hop_attri_saab.shape[3]))
#    for i in range(0, third_hop_attri_saab.shape[1], 2):
#        for j in range(0, third_hop_attri_saab.shape[2], 2):
#            pooled_attri_3[:,int(i/2),int(j/2),:] = third_hop_attri_saab[:,i,j,:]
#    print("pooled attribute 3 shape:", pooled_attri_3.shape)
    pooled_attri_3 = third_hop_attri_saab
                
    # Saab dimension reduction from 27 dim to 12 dim
    pca_params_3 = Saab_Getkernel(pooled_attri_3, "149")
    third_hop_attri = Saab_Getfeature(pooled_attri_3, pca_params_3)
                    
    print("third hop attribute shape:", third_hop_attri.shape)
    
    return pca_params_3, third_hop_attri
#%%
def LAG(positive3, negative3, cluster_num, alpha):
    
    # positive3 shape: 141987, 150
    
    # kmean 20 clusters
#    alpha = 1
#    cluster_num = [10, 5]
    km1 = KMeans(n_clusters=cluster_num[0]).fit(positive3)
    km2 = KMeans(n_clusters=cluster_num[1]).fit(negative3)
    pos_centroids = km1.cluster_centers_
    neg_centroids = km2.cluster_centers_
#    pos_km_labels = km1.labels_
#    neg_km_labels = km2.labels_
    
    target_p = np.zeros((positive3.shape[0] + negative3.shape[0], sum(cluster_num)))
    target_p_norm = np.zeros((positive3.shape[0] + negative3.shape[0], sum(cluster_num)))
    
    for k in range(positive3.shape[0]):
        for m in range(cluster_num[0]):
            target_p[k,m] = math.exp(-alpha * (np.linalg.norm(positive3[k] - pos_centroids[m])))
            
    for k in range(negative3.shape[0]):
        for m in range(cluster_num[1]):
            target_p[positive3.shape[0]+k, cluster_num[0]+m] = np.linalg.norm(negative3[k] - neg_centroids[m])
    
    for k in range(positive3.shape[0]+negative3.shape[0]):
        for m in range(sum(cluster_num)):
            target_p_norm[k,m] = target_p[k,m] / np.sum(target_p[k])
    
    dat = np.concatenate((positive3, negative3), axis=0)
    data = np.concatenate((dat, np.ones((dat.shape[0],1))), axis=1)
    print("appended data shape:", data.shape)
    
    # least square regression
    weight = scipy.linalg.lstsq(data, target_p_norm, cond=0.05)[0]
    w = weight[1:weight.shape[0]]
    c = weight[0].reshape(1,-1)
#    W_matrix = np.matmul(np.linalg.pinv(data), target_p_norm)
    
    print("Weight matrix shape:", w.shape)
    
    lag_all = np.matmul(data, w) + c
    
    lagpos3 = lag_all[:positive3.shape[0]]
    lagneg3 = lag_all[positive3.shape[0]:]
    
    return w, c ,lagpos3, lagneg3
#%%
def test_one_hop(ext_test_imgs, pca_params_1):
    
    n_samp = ext_test_imgs.shape[0]
    h = ext_test_imgs.shape[1]-8 #256
    w = ext_test_imgs.shape[2]-8 #384
    
    test_one_hop_long = np.zeros((n_samp, h,w, 27))
    
    for k in range(n_samp):
        for i in range(4, h+4):
            for j in range(4,w+4):
                top = np.concatenate((ext_test_imgs[k, i-1, j-1, :], ext_test_imgs[k,i-1,j,:], ext_test_imgs[k,i-1,j+1,:]), axis=None)
                
                middle = np.concatenate((ext_test_imgs[k, i, j-1, :], ext_test_imgs[k,i,j,:], ext_test_imgs[k,i,j+1,:]), axis=None)
                
                bottom = np.concatenate((ext_test_imgs[k, i+1, j-1, :], ext_test_imgs[k,i+1,j,:], ext_test_imgs[k,i+1,j+1,:]), axis=None)
                test_one_hop_long[k,i-4,j-4,:] = np.concatenate((top, middle, bottom), axis=None)
    
    pooled_attri = np.zeros((n_samp, int(test_one_hop_long.shape[1]/2), int(test_one_hop_long.shape[2]/2), test_one_hop_long.shape[3]))
    for i in range(0, test_one_hop_long.shape[1], 2):
        for j in range(0, test_one_hop_long.shape[2], 2):
            pooled_attri[:,int(i/2),int(j/2),:] = test_one_hop_long[:,i,j,:]
    print("pooled attribute 1 shape:", pooled_attri.shape)
                
    # Saab dimension reduction from 27 dim to 12 dim
    test_one_hop_attri = Saab_Getfeature(pooled_attri, pca_params_1)
    print("Test 1st hop attribute shape:", test_one_hop_attri.shape)
    
    return test_one_hop_attri

def test_second_hop(test_one_hop_attri, pca_params_2):
    
    n_samp = test_one_hop_attri.shape[0]
    h = test_one_hop_attri.shape[1] #256
    w = test_one_hop_attri.shape[2] #384
    
    padded_test_one_hop_attri = np.lib.pad(test_one_hop_attri, ((0,0), (2,2), (2,2), (0,0)), 'symmetric')   
    print("padded test attribute shape", padded_test_one_hop_attri.shape) # 100, 260, 388, 12
    
    test_sec_hop_long = np.zeros((n_samp, h,w, 108))
    
    for k in range(n_samp):
        for i in range(2,2+h): #2-258
            for j in range(2,2+w): #2-386
                atttotal = []
                for m in range(12):
                    top = np.concatenate((padded_test_one_hop_attri[k,i-2,j-2,m], padded_test_one_hop_attri[k,i-2,j,m], padded_test_one_hop_attri[k,i-2,j+2,m]), axis=None)
                    mid = np.concatenate((padded_test_one_hop_attri[k,i,j-2,m], padded_test_one_hop_attri[k,i,j,m], padded_test_one_hop_attri[k,i,j+2,m]), axis=None)
                    bot = np.concatenate((padded_test_one_hop_attri[k,i+2,j-2,m], padded_test_one_hop_attri[k,i+2,j,m], padded_test_one_hop_attri[k,i+2,j+2,m]), axis=None)
                    attslice = np.concatenate((top, mid, bot), axis=None) #9
                    atttotal.extend(attslice)
                atttotal = np.array(atttotal)
                test_sec_hop_long[k,i-2,j-2,:] = atttotal
                
    pooled_attri = np.zeros((n_samp, int(test_sec_hop_long.shape[1]/2), int(test_sec_hop_long.shape[2]/2), test_sec_hop_long.shape[3]))
    for i in range(0, test_sec_hop_long.shape[1], 2):
        for j in range(0, test_sec_hop_long.shape[2], 2):
            pooled_attri[:,int(i/2),int(j/2),:] = test_sec_hop_long[:,i,j,:]
    print("pooled attribute 2 shape:", pooled_attri.shape)
                
    # Saab dimension reduction from 27 dim to 12 dim
    test_sec_hop_attri = Saab_Getfeature(pooled_attri, pca_params_2)
    print("Test 2nd hop attribute shape:", test_sec_hop_attri.shape)
    
    return test_sec_hop_attri
    

def test_third_hop(test_sec_hop_attri, pca_params_3):
    
    # 8 nearest neighbors
    n_samp = test_sec_hop_attri.shape[0]
    h = test_sec_hop_attri.shape[1] #64
    w = test_sec_hop_attri.shape[2] #96
    
    #sec_hop_attri shape: 100, 64, 96, 50
    padded_test_sec_hop_attri = np.lib.pad(test_sec_hop_attri, ((0,0), (2,2), (2,2), (0,0)), 'symmetric')   
    print("padded attribute shape", padded_test_sec_hop_attri.shape) # 100, 264, 392, 50
    
    test_third_hop_long = np.zeros((n_samp, h, w, test_sec_hop_attri.shape[-1]*9))
    
    for k in range(n_samp):
        for i in range(2,2+h): #4-260
            for j in range(2,2+w): #4-388
                atttotal = []
                for m in range(test_sec_hop_attri.shape[-1]):
                    top = np.concatenate((padded_test_sec_hop_attri[k,i-2,j-2,m], padded_test_sec_hop_attri[k,i-2,j,m], padded_test_sec_hop_attri[k,i-2,j+2,m]), axis=None)
                    mid = np.concatenate((padded_test_sec_hop_attri[k,i,j-2,m], padded_test_sec_hop_attri[k,i,j,m], padded_test_sec_hop_attri[k,i,j+2,m]), axis=None)
                    bot = np.concatenate((padded_test_sec_hop_attri[k,i+2,j-2,m], padded_test_sec_hop_attri[k,i+2,j,m], padded_test_sec_hop_attri[k,i+2,j+2,m]), axis=None)
                    attslice = np.concatenate((top, mid, bot), axis=None) #9
                    atttotal.extend(attslice)
                atttotal = np.array(atttotal)
                test_third_hop_long[k,i-2,j-2,:] = atttotal
    
#    pooled_attri_3 = np.zeros((n_samp, int(test_third_hop_long.shape[1]/2), int(test_third_hop_long.shape[2]/2), test_third_hop_long.shape[3]))
#    for i in range(0, test_third_hop_long.shape[1], 2):
#        for j in range(0, test_third_hop_long.shape[2], 2):
#            pooled_attri_3[:,int(i/2),int(j/2),:] = test_third_hop_long[:,i,j,:]
#    print("pooled attribute 3 shape:", pooled_attri_3.shape)
    pooled_attri_3 = test_third_hop_long
                
    # Saab dimension reduction from 27 dim to 12 dim
    test_third_hop_attri = Saab_Getfeature(pooled_attri_3, pca_params_3)
                    
    print("test 3rd hop attribute shape:", test_third_hop_attri.shape)
    
    return test_third_hop_attri
#%%
if __name__=='__main__':
    
    start = time.time()
    train_dir_casia = "/Users/mac/Desktop/mcl_2019/image_splicing/example_files/CASIA2/"
    train_dir_colum = "/Users/mac/Desktop/mcl_2019/image_splicing/example_files/Columbia/"
    All_train_test = "/Users/mac/Desktop/mcl_2019/image_splicing/transfer_files/All_train_test/"
    
#    test_dir = '/Users/mac/Desktop/mcl_2019/image_splicing/example_files/COLUMBIA/'
    img_dir = 'img/'
    GC_dir = 'GC_ver0and1/'
    All_edge_dir = 'All_edges_0p2_ver0and1/'
    save_dir = '/Users/mac/Desktop/mcl_2019/image_splicing/Output_Text_Files/'
    save_dir_matFiles = '/Users/mac/Desktop/mcl_2019/image_splicing/Output_Mat_Files/'
    classifier = 5 # use which hop attribute

#%%    
#     save train images name into a pickle
#    filenames_casia = np.random.permutation(os.listdir(train_dir_casia + img_dir)).tolist()
#    filenames_colum = np.random.permutation(os.listdir(train_dir_casia + img_dir)).tolist()
#    # 80% of both is train, 20% of both is test
#    
#    casia_train = filenames_casia[:int(len(filenames_casia)*0.8)]
#    colum_train = filenames_colum[:int(len(filenames_colum)*0.8)]
#    train_filenames = casia_train+colum_train
#    
#    casia_test = filenames_casia[int(len(filenames_casia)*0.8):]
#    colum_test = filenames_colum[int(len(filenames_colum)*0.8):]
#    test_filenames = casia_test+colum_test
    
    
#    with open(save_dir_matFiles + 'filenames_alltrain100.pkl', 'wb') as fid:
#        cPickle.dump(train_filenames, fid)
#        
##     read train images name from pickle    
#    with open(save_dir_matFiles + 'filenames_alltest100.pkl', 'wb') as fid:
#        cPickle.dump(test_filenames, fid)
    with open(save_dir_matFiles + 'filenames_alltrain100.pkl', 'rb') as fid:
        train_filenames = cPickle.load(fid)
        
#     read train images name from pickle    
    with open(save_dir_matFiles + 'filenames_alltest100.pkl', 'rb') as fid:
        test_filenames = cPickle.load(fid)
    ext_train_imgs = boundary_extension(All_train_test+img_dir, train_filenames)
    All_edges = read_all_edge(All_train_test+All_edge_dir, train_filenames)
    GC = read_gc(All_train_test+GC_dir, train_filenames)


#%%    
    # create train_labels for training pixels: GC can be used as label
    
    pca_params_1, one_hop_attri = one_hop(ext_train_imgs)
    cPickle.dump(pca_params_1, open(save_dir_matFiles+'saab_params_1.sav', 'wb'))
    cPickle.dump(one_hop_attri, open(save_dir_matFiles+'first_hop_attributes.pkl', 'wb'))
    end = time.time()
    print("Time 1:", end-start)
    
    
    pca_params_2, second_hop_attri = second_hop(one_hop_attri)
    cPickle.dump(pca_params_2, open(save_dir_matFiles+'saab_params_2.sav', 'wb'))
    cPickle.dump(second_hop_attri, open(save_dir_matFiles+'second_hop_attributes.pkl', 'wb'))
    end2 = time.time()
    print("Time 2:", end2-end)
    #with open(save_dir_matFiles + 'second_hop_attributes.pkl', 'rb') as fid:
    #    second_hop_attri = cPickle.load(fid)
    
    pca_params_3, third_hop_attri = third_hop(second_hop_attri)
    cPickle.dump(pca_params_3, open(save_dir_matFiles+'saab_params_3.sav', 'wb'))
    cPickle.dump(third_hop_attri, open(save_dir_matFiles+'third_hop_attributes.pkl', 'wb'))
    end3 = time.time()
    print("Time 3:", end3-end2)
    
#%%
#    one_hop_attri = cPickle.load(open(save_dir_matFiles+'first_hop_attributes.pkl', 'rb'))
#    second_hop_attri = cPickle.load(open(save_dir_matFiles+'second_hop_attributes.pkl', 'rb'))
#    third_hop_attri = cPickle.load(open(save_dir_matFiles+'third_hop_attributes.pkl', 'rb'))
#    print(one_hop_attri.shape)
    # select same number of positive and negative pixels
    # positive: pixels along the spliced boundary
    # negative: randomly authentic edge pixels
   
    # bilinear interpolation
    hop1feat_interp = np.zeros((GC.shape[0], GC.shape[1], GC.shape[2], one_hop_attri.shape[-1]))
    hop2feat_interp = np.zeros((GC.shape[0], GC.shape[1], GC.shape[2], second_hop_attri.shape[-1]))
    hop3feat_interp = np.zeros((GC.shape[0], GC.shape[1], GC.shape[2], third_hop_attri.shape[-1]))

    for k in range(one_hop_attri.shape[0]):
        hop1feat_interp[k] = cv2.resize(one_hop_attri[k], (GC.shape[2], GC.shape[1]), interpolation = cv2.INTER_LINEAR)
       
        hop2feat_interp[k] = cv2.resize(second_hop_attri[k], (GC.shape[2], GC.shape[1]), interpolation = cv2.INTER_CUBIC)
       
        hop3feat_interp[k] = cv2.resize(third_hop_attri[k], (GC.shape[2], GC.shape[1]), interpolation = cv2.INTER_CUBIC)
   
    cPickle.dump(hop1feat_interp, open(save_dir_matFiles+'interp_hop1feat.pkl', 'wb'))
    np.save(save_dir_matFiles+'interp_hop2feat.pkl', hop2feat_interp)
    np.save(save_dir_matFiles+'interp_hop3feat.npy', hop3feat_interp)
#%%    

    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((10,10), np.uint8)
    boundary = np.zeros((GC.shape), dtype=int)
    authentic_edges = np.zeros((GC.shape), dtype=int)
    
    for k in range(GC.shape[0]):
        Splicing_edge_image = RobertsAlogrithm(GC[k])  # very thin
        
        for i in range(GC.shape[1]):
            for j in range(GC.shape[2]):
                if Splicing_edge_image[i][j] != 0:
                    Splicing_edge_image[i][j] = 255
        
        Splicing_edge_image1 = cv2.dilate(Splicing_edge_image,kernel1,iterations = 1)
        
        boundary[k] = Splicing_edge_image1
        
        Splicing_edge_image2 = cv2.dilate(Splicing_edge_image,kernel2,iterations = 1)
        
        Authetic_edge_image = np.zeros(GC[k].shape)  # not very clean&thin
        for i in range(GC.shape[1]):
            for j in range(GC.shape[2]):
                if All_edges[k][i][j] != 0 and Splicing_edge_image2[i][j] == 0:
                    Authetic_edge_image[i][j] = 255
                    
        authentic_edges[k] = Authetic_edge_image
#%%
    hop1feat_interp = cPickle.load(open(save_dir_matFiles+'interp_hop1feat.pkl', 'rb'))
    hop2feat_interp = np.load(save_dir_matFiles+'interp_hop2feat.npy')
    hop3feat_interp = np.load(save_dir_matFiles+'interp_hop3feat.npy')
    
#    feat_ensem = np.concatenate((hop1feat_interp, hop2feat_interp, hop3feat_interp), axis=3)
#    print("ensemble attributes:", feat_ensem.shape)
   
    positive3 = []
    negative3 = []
    positive2 = []
    negative2 = []
    positive1 = []
    negative1 = []
#    positive_ensem = []
#    negative_ensem = []
    counts_pos = np.zeros((GC.shape[0],), dtype=int)
   
    for k in range(boundary.shape[0]):
        count = 0
        for i in range(boundary.shape[1]):
            for j in range(boundary.shape[2]):
                if boundary[k,i,j] != 0 and All_edges[k,i,j]!=0:   # spliced boundary pixel
                    count = count + 1
                    pos3 = hop3feat_interp[k,i,j]
                    positive3.append(pos3)
                    pos2 = hop2feat_interp[k,i,j]
                    positive2.append(pos2)                    
                    pos1 = hop1feat_interp[k,i,j]
                    positive1.append(pos1)
#                    pos_ensem = feat_ensem[k,i,j]
#                    positive_ensem.append(pos_ensem) # shape: number of positive pixels, 150
        
        counts_pos[k] = count          
    print("total number of positive pixels:", np.sum(counts_pos))    
    positive3 = np.array(positive3)
    positive2 = np.array(positive2)
    positive1 = np.array(positive1)
#    positive_ensem = np.array(positive_ensem)
    np.save(save_dir_matFiles+'positive_hop3_feat.npy', positive3)  
    np.save(save_dir_matFiles+'positive_hop2_feat.npy', positive2)           
    np.save(save_dir_matFiles+'positive_hop1_feat.npy', positive1)           
         
   
#%%
#    a = []
#    b = []
#    for i in range(boundary.shape[1]):
#        a.append(i)
#    for j in range(boundary.shape[2]):
#        b.append(j)
#    random.shuffle(a)
#    random.shuffle(b)
#    np.save(save_dir_matFiles+'a.npy', a)
#    np.save(save_dir_matFiles+'b.npy', b)
    
    a = np.load(save_dir_matFiles+'a.npy')
    b = np.load(save_dir_matFiles+'b.npy')
    
    
    
    
    for k in range(boundary.shape[0]):
        t = 0
        for i in a:
            for j in b:
                if (authentic_edges[k,i,j] !=0) and All_edges[k,i,j]!=0: # authentic edge pixel
                    t = t+1
                    if t <= counts_pos[k]:
                        neg3 = hop3feat_interp[k,i,j]
                        neg2 = hop2feat_interp[k,i,j]
                        neg1 = hop1feat_interp[k,i,j]
                        negative3.append(neg3)
                        negative2.append(neg2)
                        negative1.append(neg1)

                   
    negative3 = np.array(negative3)
    negative2 = np.array(negative2)
    negative1 = np.array(negative1)
    
    cPickle.dump(negative3, open(save_dir_matFiles+'negative_hop3_feat.pkl', 'wb'))
    cPickle.dump(negative2, open(save_dir_matFiles+'negative_hop2_feat.pkl', 'wb'))
    cPickle.dump(negative1, open(save_dir_matFiles+'negative_hop1_feat.pkl', 'wb'))


#%%
    positive3 = np.load(save_dir_matFiles+'positive_hop3_feat.npy')
    negative3 = cPickle.load(open(save_dir_matFiles+'negative_hop3_feat.pkl', 'rb'))

    cluster_num = [10, 5]
    alpha = 1
    
    w, c, lagpos3, lagneg3 = LAG(positive3, negative3, cluster_num, alpha)
    print("LAG on hop3 shape:", lagpos3.shape, lagneg3.shape)


#%%    
    positive2 = np.load(save_dir_matFiles+'positive_hop2_feat.npy')
    negative2 = cPickle.load(open(save_dir_matFiles+'negative_hop2_feat.pkl', 'rb'))
    positive1 = np.load(save_dir_matFiles+'positive_hop1_feat.npy')
    negative1 = cPickle.load(open(save_dir_matFiles+'negative_hop1_feat.pkl', 'rb'))
    # train classifiers random forest classifier
    # label: GC, data: third hop attribute
    print("-------TRAINING CLASSIFIERS-------")
    RF = False
#    second_hop_attri = cPickle.load(open(save_dir_matFiles+'second_hop_attributes.pkl', 'rb'))
    min_samples_split=2
    max_features='auto'
    bootstrap =True
    max_depth=None
    min_samples_leaf=1
    n_estimators_rf = 10

    
    if RF:
        clf1 = RandomForestClassifier(n_jobs=int(multiprocessing.cpu_count()/2), verbose=0, class_weight='balanced',
                                      n_estimators=n_estimators_rf, min_samples_split = min_samples_split,
                                      max_features = max_features, bootstrap = bootstrap,
                                      max_depth = max_depth, min_samples_leaf = min_samples_leaf)
    else:   
        clf1 = svm.SVC(probability=False,gamma=2.0,C=1.0,class_weight='balanced')
    
    pos_label = np.ones((positive3.shape[0],), dtype=int)
    neg_label = np.zeros((negative3.shape[0],), dtype=int)
    labels = np.concatenate((pos_label, neg_label), axis=0)
    print("label shape:", labels.shape)
    
    if classifier == 1: # use 1 hop attribute only

        data = np.concatenate((positive1, negative1), axis=0)
        clf1.fit(data, labels)
        predicted_train_label = clf1.predict(data)
        
        C_train = confusion_matrix(labels, predicted_train_label)
        
    elif classifier == 2: # use 2 hop attribute only
        
        data = np.concatenate((positive2, negative2), axis=0)    
        clf1.fit(data, labels)
        predicted_train_label = clf1.predict(data)
        
        C_train = confusion_matrix(labels, predicted_train_label)
        
    elif classifier == 3: # use 3 hop attribute only
        data = np.concatenate((positive3, negative3), axis=0)
        clf1.fit(data, labels)
        predicted_train_label = clf1.predict(data)
        
        C_train = confusion_matrix(labels, predicted_train_label)
        
    elif classifier == 4: # use 123 hop attribute
        
        positive_ensem = np.concatenate((positive1, positive2, positive3), axis=1)
        negative_ensem = np.concatenate((negative1, negative2, negative3), axis=1)
        data = np.concatenate((positive_ensem, negative_ensem), axis=0)
        
        clf1.fit(data, labels)
        predicted_train_label = clf1.predict(data)
        
        C_train = confusion_matrix(labels, predicted_train_label)
        
    elif classifier == 5: # use lag 3 attribute
        data = np.concatenate((lagpos3, lagneg3), axis=0)
        clf1.fit(data, labels)
        predicted_train_label = clf1.predict(data)
        
        C_train = confusion_matrix(labels, predicted_train_label)
        
    per_class_accuracy_train = np.diag(C_train.astype(np.float32))/np.sum(C_train.astype(np.float32),axis=1)
    print(per_class_accuracy_train)
#%%
        
    print("--------TEST PROCESS--------")   
    
    # read test images
#    test_filenames = np.random.permutation(os.listdir(test_dir + img_dir)).tolist()
#    with open(save_dir_matFiles + 'filenames_Columbia_test10.pkl', 'wb') as fid:
#        cPickle.dump(filenames, fid)
        
#    with open(save_dir_matFiles + 'filenames_Columbia_test10.pkl', 'rb') as fid:
#        test_filenames = cPickle.load(fid)
        
    ext_test_imgs = boundary_extension(All_train_test+img_dir, test_filenames)
    test_GC = read_gc(All_train_test+GC_dir, test_filenames)
    test_All_edge = read_all_edge(All_train_test+All_edge_dir, test_filenames)
    
    # attributes
#    pca_params_1 = cPickle.load(open(save_dir_matFiles+'saab_params_1.sav', 'rb'))
#    pca_params_2 = cPickle.load(open(save_dir_matFiles+'saab_params_2.sav', 'rb'))
#    pca_params_3 = cPickle.load(open(save_dir_matFiles+'saab_params_3.sav', 'rb'))

    test_one_hop_attri = test_one_hop(ext_test_imgs, pca_params_1)
    cPickle.dump(test_one_hop_attri, open(save_dir_matFiles+'test_1_hop_attribute.pkl', 'wb'))
    
    test_sec_hop_attri = test_second_hop(test_one_hop_attri, pca_params_2)
    cPickle.dump(test_sec_hop_attri, open(save_dir_matFiles+'test_2_hop_attribute.pkl', 'wb'))
    
    test_third_hop_attri = test_third_hop(test_sec_hop_attri, pca_params_3)
    cPickle.dump(test_third_hop_attri, open(save_dir_matFiles+'test_3_hop_attribute.pkl', 'wb'))
    
#%%
    # bilinear interpolation
    test_hop1feat_interp = np.zeros((test_GC.shape[0], test_GC.shape[1], test_GC.shape[2], test_one_hop_attri.shape[-1]))
    test_hop2feat_interp = np.zeros((test_GC.shape[0], test_GC.shape[1], test_GC.shape[2], test_sec_hop_attri.shape[-1]))
    test_hop3feat_interp = np.zeros((test_GC.shape[0], test_GC.shape[1], test_GC.shape[2], test_third_hop_attri.shape[-1]))

    for k in range(test_one_hop_attri.shape[0]):
        test_hop1feat_interp[k] = cv2.resize(test_one_hop_attri[k], (test_GC.shape[2], test_GC.shape[1]), interpolation = cv2.INTER_LINEAR)
       
        test_hop2feat_interp[k] = cv2.resize(test_sec_hop_attri[k], (test_GC.shape[2], test_GC.shape[1]), interpolation = cv2.INTER_CUBIC)
       
        test_hop3feat_interp[k] = cv2.resize(test_third_hop_attri[k], (test_GC.shape[2], test_GC.shape[1]), interpolation = cv2.INTER_CUBIC)
    
    cPickle.dump(test_hop1feat_interp, open(save_dir_matFiles+'test_interp_hop1feat.pkl', 'wb'))
    cPickle.dump(test_hop2feat_interp, open(save_dir_matFiles+'test_interp_hop2feat.pkl', 'wb'))
    np.save(save_dir_matFiles+'test_interp_hop3feat', test_hop3feat_interp)

#%% 
    
#    test_hop1feat_interp = cPickle.load(open(save_dir_matFiles+'test_interp_hop1feat.pkl', 'rb'))
#    test_hop2feat_interp = cPickle.load(open(save_dir_matFiles+'test_interp_hop2feat.pkl', 'rb'))
#    test_hop3feat_interp = cPickle.load(open(save_dir_matFiles+'test_interp_hop3feat.pkl', 'rb'))
        
       
    test_boundary = np.zeros((test_GC.shape), dtype=int)
    test_authentic_edges = np.zeros((test_GC.shape), dtype=int)
    
    for k in range(test_GC.shape[0]):
        test_Splicing_edge_image = RobertsAlogrithm(test_GC[k])  # very thin
        
        for i in range(test_GC.shape[1]):
            for j in range(test_GC.shape[2]):
                if test_Splicing_edge_image[i][j] != 0:
                    test_Splicing_edge_image[i][j] = 255
        
        test_Splicing_edge_image1 = cv2.dilate(test_Splicing_edge_image,kernel1,iterations = 1)
        
        test_boundary[k] = test_Splicing_edge_image1
        
        test_Splicing_edge_image2 = cv2.dilate(test_Splicing_edge_image,kernel2,iterations = 1)
        
        test_Authetic_edge_image = np.zeros(test_GC[k].shape)  # not very clean&thin
        for i in range(test_GC.shape[1]):
            for j in range(test_GC.shape[2]):
                if test_All_edge[k][i][j] != 0 and test_Splicing_edge_image2[i][j] == 0:
                    test_Authetic_edge_image[i][j] = 255
                    
        test_authentic_edges[k] = test_Authetic_edge_image
    
#%%    
    test_positive3 = []
    test_negative3 = []
    test_positive2 = []
    test_negative2 = []
    test_positive1 = []
    test_negative1 = []
#    positive_ensem = []
#    negative_ensem = []
   
    for k in range(test_boundary.shape[0]):
        for i in range(test_boundary.shape[1]):
            for j in range(boundary.shape[2]):
                if test_All_edge[k,i,j] != 0 and test_boundary[k,i,j] != 0:   # spliced boundary pixel
                    
                    test_pos3 = test_hop3feat_interp[k,i,j]
                    test_positive3.append(test_pos3)
                    test_pos2 = test_hop2feat_interp[k,i,j]
                    test_positive2.append(test_pos2)                    
                    test_pos1 = test_hop1feat_interp[k,i,j]
                    test_positive1.append(test_pos1)
                    

                if test_All_edge[k,i,j] != 0 and test_authentic_edges[k,i,j] != 0:   # authentic boundary pixel
                    
                    test_neg3 = test_hop3feat_interp[k,i,j]
                    test_negative3.append(test_neg3)
                    test_neg2 = test_hop2feat_interp[k,i,j]
                    test_negative2.append(test_neg2)                    
                    test_neg1 = test_hop1feat_interp[k,i,j]
                    test_negative1.append(test_neg1)

                 
    test_positive3 = np.array(test_positive3)
    test_positive2 = np.array(test_positive2)
    test_positive1 = np.array(test_positive1)
    test_negative3 = np.array(test_negative3)
    test_negative2 = np.array(test_negative2)
    test_negative1 = np.array(test_negative1)
    
    np.save(save_dir_matFiles+'test_positive_hop3_feat.npy', test_positive3)  
    np.save(save_dir_matFiles+'test_positive_hop2_feat.npy', test_positive2)           
    np.save(save_dir_matFiles+'test_positive_hop1_feat.npy', test_positive1)           
    np.save(save_dir_matFiles+'test_negative_hop3_feat.npy', test_negative3)  
    np.save(save_dir_matFiles+'test_negative_hop2_feat.npy', test_negative2)           
    np.save(save_dir_matFiles+'test_negative_hop1_feat.npy', test_negative1)    

    
#%%    
    test_positive1 = np.load(save_dir_matFiles+'test_positive_hop1_feat.npy')
    test_positive2 = np.load(save_dir_matFiles+'test_positive_hop2_feat.npy')
    test_positive3 = np.load(save_dir_matFiles+'test_positive_hop3_feat.npy')
    test_negative1 = np.load(save_dir_matFiles+'test_negative_hop1_feat.npy')
    test_negative2 = np.load(save_dir_matFiles+'test_negative_hop2_feat.npy')
    test_negative3 = np.load(save_dir_matFiles+'test_negative_hop3_feat.npy')
    
#    test_lagpos3 = test_positive3 * W + c
#    test_lagneg3 = test_negative3 * W + c
     
    test_pos_labels = np.ones((test_positive1.shape[0],))
    test_neg_labels = np.zeros((test_negative1.shape[0],))
    test_true_labels = np.concatenate((test_pos_labels, test_neg_labels), axis=0)
    
    # classify by trained classifiers
    if classifier == 1:  # use hop 1
        test_data = np.concatenate((test_positive1, test_negative1), axis=0)
        test_predicted_label = clf1.predict(test_data)
        C_test = confusion_matrix(test_true_labels, test_predicted_label)
        
    elif classifier == 2:  # use hop 2
        test_data = np.concatenate((test_positive2, test_negative2), axis=0)
        test_predicted_label = clf1.predict(test_data)
        C_test = confusion_matrix(test_true_labels, test_predicted_label)     
    
    elif classifier == 3:  # use hop 3
        test_data = np.concatenate((test_positive3, test_negative3), axis=0)
        test_predicted_label = clf1.predict(test_data)
        C_test = confusion_matrix(test_true_labels, test_predicted_label)  
        
    elif classifier == 4:  # use hop 123
        test_positive_ensem = np.concatenate((test_positive1, test_positive2, test_positive3), axis=1)
        test_negative_ensem = np.concatenate((test_negative1, test_negative2, test_negative3), axis=1)
        test_data = np.concatenate((test_positive_ensem, test_negative_ensem), axis=0)
        
        test_predicted_label = clf1.predict(test_data)
        C_test = confusion_matrix(test_true_labels, test_predicted_label)
        
    elif classifier == 5:
        test_data = np.concatenate((test_lagpos3, test_lagneg3), axis=0)
        test_predicted_label = clf1.predict(test_data)
        C_test = confusion_matrix(test_true_labels, test_predicted_label)          
        
    per_class_accuracy_test = np.diag(C_test.astype(np.float32))/np.sum(C_test.astype(np.float32),axis=1)
    print(per_class_accuracy_test)
