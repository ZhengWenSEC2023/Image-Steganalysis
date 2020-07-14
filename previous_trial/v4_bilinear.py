#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:19:51 2019

@author: Yao Zhu
"""

import numpy as np
import _pickle as cPickle
from skimage import io

from matplotlib import pyplot as plt
plt.interactive(True)
from PCA import saab
import time
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import multiprocessing
from sklearn.metrics import confusion_matrix
import math
import cv2
from previous_trial.LAG import LAG_Unit

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
#         padded_img_samp = np.lib.pad(img_samp, ((1,1),(1,1),(0,0)), 'symmetric') # symmetric padding
#        print(padded_img_samp.shape)
        extended_imgs.append(img_samp)
        
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
    train_images = train_images[:60000]
    train_labels = np.ones((train_images.shape[0],))
    kernel_sizes = str(train_images.shape[1])

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

def concatenate(attributes, k, stride):

    attribute = attributes[k]
    # n_samp = attribute.shape[0]
    h = attribute.shape[0]
    w = attribute.shape[1]
    spec = attribute.shape[2]

    padded_attribute  = np.lib.pad(attribute, ((stride, stride), (stride,stride), (0,0)), 'symmetric')
    # print("padded attribute shape:", padded_attribute.shape)

    attribute_saab = np.zeros((h, w, spec * 9))
    # for k in range(n_samp):
    for i in range(stride, stride+h):
        for j in range(stride, stride+w):
            concate = np.zeros((9*spec,))
            for m in range(spec):
                concate[9*m: 9*m+9] = np.concatenate((padded_attribute[i-stride, j-stride, m],
                                                      padded_attribute[i-stride, j, m],
                                                      padded_attribute[i-stride, j+stride, m],
                                                      padded_attribute[i, j-stride, m],
                                                      padded_attribute[i, j, m],
                                                      padded_attribute[i, j+stride, m],
                                                      padded_attribute[i+stride, j-stride, m],
                                                      padded_attribute[i+stride, j, m],
                                                      padded_attribute[i+stride, j+stride, m]), axis=None)
            attribute_saab[i-stride, j-stride, :] = concate

    return attribute_saab

def select_samples(GCs, All_edges, kernel1, kernel2, k):

    GC = GCs[k]  # 256, 384
    All_edge = All_edges[k]  # 256, 384

    Splicing_edge_image = RobertsAlogrithm(GC)  # very thin

    for i in range(GC.shape[0]):
        for j in range(GC.shape[1]):
            if Splicing_edge_image[i][j] != 0:
                Splicing_edge_image[i][j] = 255

    Splicing_edge_image1 = cv2.dilate(Splicing_edge_image, kernel1, iterations=1)  # spliced

    Splicing_edge_image2 = cv2.dilate(Splicing_edge_image, kernel2, iterations=1)

    Authetic_edge_image = np.zeros(GC.shape) # authentic

    for i in range(GC.shape[0]):
        for j in range(GC.shape[1]):
            if All_edge[i][j] != 0 and Splicing_edge_image2[i][j] == 0:
                Authetic_edge_image[i][j] = 255

    return Splicing_edge_image1, Authetic_edge_image

def pooling(attribute):

    # Pooling from 100,256,384,27 to 100,128,192,27
    # block_reduce(one_hop_attri_saab, block_size=(4,4,1), func=np.mean)
    n_samp = attribute.shape[0]

    pooled_attri = np.zeros((n_samp, int(attribute.shape[1]/2), int(attribute.shape[2]/2), attribute.shape[3]))
    for i in range(0, attribute.shape[1], 2):
       for j in range(0, attribute.shape[2], 2):
           pooled_attri[:,int(i/2),int(j/2),:] = attribute[:,i,j,:]

    return pooled_attri


def one_hop(extended_imgs):

    # # 8 nearest neighbors
    a = time.time()
    n_samp = extended_imgs.shape[0]

    num_cores = int(multiprocessing.cpu_count()/2)

    one_hop_attri_saab = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(concatenate)(extended_imgs, k, 1) for k in range(n_samp))
    print(type(one_hop_attri_saab))

    one_hop_attri_saab = np.array(one_hop_attri_saab)
    print(one_hop_attri_saab.shape)

    b = time.time()
    print("time for concatenate 1st :", b-a)

    # Saab dimension reduction from 27 dim to 12 dim
    pca_params_1 = Saab_Getkernel(one_hop_attri_saab, "11")
    c = time.time()
    print("time for kernel:", c-b)

    one_hop_attri = Saab_Getfeature(one_hop_attri_saab, pca_params_1)
    d = time.time()
    print("time for feature:", d-c)


    print("one hop attribute shape:", one_hop_attri.shape)


    return pca_params_1, one_hop_attri


def second_hop(one_hop_attri):

    a = time.time()
    num_cores = int(multiprocessing.cpu_count() / 2)
    n_samp = one_hop_attri.shape[0]

    sec_hop_attri_saab = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(concatenate)(one_hop_attri, k, 2) for k in range(n_samp))
    print(type(sec_hop_attri_saab))

    sec_hop_attri_saab = np.array(sec_hop_attri_saab)
    print(sec_hop_attri_saab.shape)

    b = time.time()
    print("time for concatenate 2nd :", b - a)
                
    # Saab dimension reduction from 27 dim to 12 dim
    pca_params_2 = Saab_Getkernel(sec_hop_attri_saab, "47")
    c = time.time()
    print("time for kernel:", c-b)

    sec_hop_attri = Saab_Getfeature(sec_hop_attri_saab, pca_params_2)
    d = time.time()
    print("time for feature:", d-b)

    print("second hop attribute shape:", sec_hop_attri.shape)
    
    return pca_params_2, sec_hop_attri


def third_hop(sec_hop_attri):

    a = time.time()
    n_samp = sec_hop_attri.shape[0]

    num_cores = int(multiprocessing.cpu_count() / 2)

    third_hop_attri_saab = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(concatenate)(sec_hop_attri, k, 3) for k in range(n_samp))
    print(type(third_hop_attri_saab))

    b = time.time()
    print("time for concatenate 3rd :", b - a)

    third_hop_attri_saab = np.array(third_hop_attri_saab)
    print(third_hop_attri_saab.shape)

    # Saab dimension reduction from 27 dim to 12 dim
    pca_params_3 = Saab_Getkernel(third_hop_attri_saab, "146")
    c = time.time()
    print("time for kernel:", c-b)

    third_hop_attri = Saab_Getfeature(third_hop_attri_saab, pca_params_3)
    d = time.time()
    print("time for feature:", d-c)

    print("third hop attribute shape:", third_hop_attri.shape)
    
    return pca_params_3, third_hop_attri

#%%
def test_one_hop(ext_test_imgs, pca_params_1):

    a = time.time()
    n_samp = ext_test_imgs.shape[0]
    num_cores = int(multiprocessing.cpu_count() / 2)

    test_one_hop_attri_saab = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(concatenate)(ext_test_imgs, k, 1) for k in range(n_samp))
    print(type(test_one_hop_attri_saab))
    test_one_hop_attri_saab = np.array(test_one_hop_attri_saab)
    print(test_one_hop_attri_saab.shape)
    b = time.time()
    print("time for concatenate 1st test:", b - a)
                
    # Saab dimension reduction from 27 dim to 12 dim
    test_one_hop_attri = Saab_Getfeature(test_one_hop_attri_saab, pca_params_1)
    print("Test 1st hop attribute shape:", test_one_hop_attri.shape)
    
    return test_one_hop_attri

def test_second_hop(test_one_hop_attri, pca_params_2):

    a = time.time()
    n_samp = test_one_hop_attri.shape[0]
    num_cores = int(multiprocessing.cpu_count() / 2)

    test_sec_hop_attri_saab = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(concatenate)(test_one_hop_attri, k, 2) for k in range(n_samp))
    print(type(test_sec_hop_attri_saab))
    test_sec_hop_attri_saab = np.array(test_sec_hop_attri_saab)
    print(test_sec_hop_attri_saab.shape)
    b = time.time()
    print("time for concatenate 2nd :", b - a)
                
    # Saab dimension reduction from 27 dim to 12 dim
    test_sec_hop_attri = Saab_Getfeature(test_sec_hop_attri_saab, pca_params_2)
    print("Test 2nd hop attribute shape:", test_sec_hop_attri.shape)
    
    return test_sec_hop_attri
    

def test_third_hop(test_sec_hop_attri, pca_params_3):
    
    # 8 nearest neighbors
    a = time.time()
    n_samp = test_sec_hop_attri.shape[0]
    num_cores = int(multiprocessing.cpu_count() / 2)

    test_third_hop_attri_saab = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(concatenate)(test_sec_hop_attri, k, 3) for k in range(n_samp))
    print(type(test_third_hop_attri_saab))
    test_third_hop_attri_saab = np.array(test_third_hop_attri_saab)
    print(test_third_hop_attri_saab.shape)
    b = time.time()
    print("time for concatenate 3rd :", b - a)
                
    # Saab dimension reduction from 27 dim to 12 dim
    test_third_hop_attri = Saab_Getfeature(test_third_hop_attri_saab, pca_params_3)
                    
    print("test 3rd hop attribute shape:", test_third_hop_attri.shape)
    
    return test_third_hop_attri


def output_map(predicted_label, pos_count, neg_count, positive1):
    t1 = 0
    n_samp = len(pos_count)
    total_pos_num = positive1.shape[0]
    output = np.zeros((n_samp, 256, 384))

    for k in range(n_samp):
        num_pos_samp_k = len(pos_count[k])

        for m in range(num_pos_samp_k):
            pos_count[k][m][2] = predicted_label[t1 + m]

        t1 = t1 + num_pos_samp_k

    t2 = 0
    for k in range(n_samp):
        num_neg_samp_k = len(neg_count[k])

        for m in range(num_neg_samp_k):
            neg_count[k][m][2] = predicted_label[total_pos_num + t2 + m]

        t2 = t2 + num_neg_samp_k

    for k in range(n_samp):
        num_pos_samp_k = len(pos_count[k])
        num_neg_samp_k = len(neg_count[k])

        for m in range(num_pos_samp_k):
            output[k, pos_count[k][m][0], pos_count[k][m][1]] = pos_count[k][m][2]

        for n in range(num_neg_samp_k):
            output[k, neg_count[k][n][0], neg_count[k][n][1]] = neg_count[k][n][2]

    return output

#%%
if __name__=='__main__':

    # train_dir_casia = "/Users/mac/Desktop/mcl_2019/image_splicing/example_files/CASIA2/"
    train_dir_casia = "/home/yzhu/image_splicing/example_files/CASIA2/"
    train_dir_colum = "/home/yzhu/image_splicing/example_files/Columbia/"
    # train_dir_colum = "/Users/mac/Desktop/mcl_2019/image_splicing/example_files/Columbia/"

    All_train_test = "/mnt/yaozhu/image_splicing_mnt/example_files/All_train_test/"
    
#    test_dir = '/Users/mac/Desktop/mcl_2019/image_splicing/example_files/COLUMBIA/'
    img_dir = 'img/'
    GC_dir = 'GC_ver0and1/'
    All_edge_dir = 'All_edges_0p2_ver0and1/'
    # save_dir = '/mnt/yaozhu/image_splicing_mnt/Output_Text_Files/'
    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/100/'

    classifier = 4  # use which hop attribute
    print("classifier used:", classifier)

#%%    
    # # save train images name into a pickle
    # filenames_casia = np.random.permutation(os.listdir(train_dir_casia + img_dir)).tolist()
    # filenames_colum = np.random.permutation(os.listdir(train_dir_colum + img_dir)).tolist()
    #
    # # use how many number of casia, columbia
    # filenames_casia_used = filenames_casia[:100]
    # filenames_colum_used = filenames_colum[:100]
    # print("how many casia images used:", len(filenames_casia_used))
    # print("how many columbia images used:", len(filenames_colum_used))
    #
    # all_filenames = filenames_casia_used + filenames_colum_used
    # random.shuffle(all_filenames)
    # random.shuffle(all_filenames)
    #
    # print(len(all_filenames))
    #
    # # 80% of both is train, 20% of both is test
    # train_filenames = all_filenames[:int(len(all_filenames)*0.8)]
    # test_filenames = all_filenames[int(len(all_filenames)*0.8):]
    # print(len(train_filenames), len(test_filenames))
    #
    # # train_filenames = filenames_casia_used
    # # test_filenames = filenames_colum_used
    #
    # with open(save_dir_matFiles + 'filenames_alltrain100.pkl', 'wb') as fid:
    #    cPickle.dump(train_filenames, fid)
    #
    # with open(save_dir_matFiles + 'filenames_alltest100.pkl', 'wb') as fid:
    #    cPickle.dump(test_filenames, fid)

    # with open(save_dir_matFiles + 'filenames_alltrain100.pkl', 'rb') as fid:
    #     train_filenames = cPickle.load(fid)
    #
    # with open(save_dir_matFiles + 'filenames_alltest100.pkl', 'rb') as fid:
    #     test_filenames = cPickle.load(fid)

    # ext_train_imgs = boundary_extension(All_train_test+img_dir, train_filenames)
    # print("extended images shape:", ext_train_imgs.shape)
    #
    # All_edges = read_all_edge(All_train_test+All_edge_dir, train_filenames)
    # print("All edges shape:", All_edges.shape)
    #
    # GC = read_gc(All_train_test+GC_dir, train_filenames)
    # print("GC shape:", GC.shape)

    # del filenames_casia, filenames_casia_used, filenames_colum, filenames_colum_used
    # del ext_train_imgs_casia, ext_train_imgs_colum, All_edges_casia, All_edges_colum, GC_casia, GC_colum

#%%
    # start = time.time()
    # pca_params_1, one_hop_attri = one_hop(ext_train_imgs)
    # cPickle.dump(pca_params_1, open(save_dir_matFiles + 'saab_params_1_100.sav', 'wb'))
    #
    # h5f = h5py.File(save_dir_matFiles+'train_1_100.h5', 'w')
    # h5f.create_dataset('attribute', data=one_hop_attri)
    # h5f.close()
    #
    # # h5f = h5py.File(save_dir_matFiles+'train_1_100.h5', 'r')
    # # one_hop_attri = h5f['attribute'][:]
    # # print(one_hop_attri.shape)
    #
    # end = time.time()
    # print("Time 1:", end - start)
    #
    # pca_params_2, second_hop_attri = second_hop(one_hop_attri)
    # cPickle.dump(pca_params_2, open(save_dir_matFiles+'saab_params_2_100.sav', 'wb'))
    #
    # h5f = h5py.File(save_dir_matFiles+'train_2_100.h5', 'w')
    # h5f.create_dataset('attribute', data = second_hop_attri)
    # h5f.close()
    #
    # # h5f = h5py.File(save_dir_matFiles+'train_2_100.h5', 'r')
    # # second_hop_attri = h5f['attribute'][:]
    # # print(second_hop_attri.shape)
    #
    # end2 = time.time()
    # print("Time 2:", end2-end)
    #
    # #pooled 2nd attri, to learn 3rd attri
    # pooled_sec_attri = pooling(second_hop_attri)
    #
    # pca_params_3, third_hop_attri = third_hop(second_hop_attri)
    #
    # cPickle.dump(pca_params_3, open(save_dir_matFiles + 'saab_params_3_100.sav', 'wb'))
    #
    # h5f = h5py.File(save_dir_matFiles+'train_3_100.h5', 'w')
    # h5f.create_dataset('attribute', data=third_hop_attri)
    # h5f.close()
    #
    # # h5f = h5py.File(save_dir_matFiles+'train_3_100.h5', 'r')
    # # third_hop_attri = h5f['attribute'][:]
    #
    # end3 = time.time()
    # print("Time 3:", end3-end2)
    #
    # # third_hop_attri = np.zeros((GC.shape[0], GC.shape[1], GC.shape[2], third_hop_attri_reduced.shape[-1]))
    # # for k in range(third_hop_attri_reduced.shape[0]):
    # #     third_hop_attri[k] = cv2.resize(third_hop_attri_reduced[k], (GC.shape[2], GC.shape[1]), interpolation=cv2.INTER_LINEAR)
    #
    # print(third_hop_attri.shape)

#%%

    # start = time.time()
    # kernel1 = np.ones((3, 3), np.uint8)
    # kernel2 = np.ones((10,10), np.uint8)
    #
    # num_cores = int(multiprocessing.cpu_count() / 2)
    # n_samp = GC.shape[0]
    #
    # spliced_authentic = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(select_samples)(GC, All_edges, kernel1, kernel2, k) for k in range(n_samp))
    #
    # print(len(spliced_authentic))
    # spliced_authentic = np.array(spliced_authentic)
    # print(spliced_authentic.shape)
    # boundary = spliced_authentic[:, 0, :, :]
    # authentic_edges = spliced_authentic[:, 1, :, :]
    #
    # print("spliced mask shape:", boundary.shape)
    # print("authentic mask shape:", authentic_edges.shape)
    #
    # end = time.time()
    # print("time for selecting:", end-start)


#%%
    # print(one_hop_attri.shape)
    # print(second_hop_attri.shape)
    # print(third_hop_attri.shape)
    #
    # positive3 = []
    # negative3 = []
    # positive2 = []
    # negative2 = []
    # positive1 = []
    # negative1 = []
    # counts_pos = np.zeros((GC.shape[0],), dtype=int)
    #
    # for k in range(boundary.shape[0]):
    #     count = 0
    #     for i in range(boundary.shape[1]):
    #         for j in range(boundary.shape[2]):
    #             if boundary[k,i,j] != 0 and All_edges[k,i,j]!=0:   # spliced boundary pixel
    #                 count = count + 1
    #                 pos3 = third_hop_attri[k,i,j]
    #                 positive3.append(pos3)
    #                 pos2 = second_hop_attri[k,i,j]
    #                 positive2.append(pos2)
    #                 pos1 = one_hop_attri[k,i,j]
    #                 positive1.append(pos1)
    #     counts_pos[k] = count
    #
    # print("total number of positive pixels:", np.sum(counts_pos))
    # positive3 = np.array(positive3)
    # positive2 = np.array(positive2)
    # positive1 = np.array(positive1)
    # np.save(save_dir_matFiles+'pos_3_100.npy', positive3)
    # np.save(save_dir_matFiles+'pos_2_100.npy', positive2)
    # np.save(save_dir_matFiles+'pos_1_100.npy', positive1)

#%%
   # a = []
   # b = []
   # for i in range(boundary.shape[1]):
   #     a.append(i)
   # for j in range(boundary.shape[2]):
   #     b.append(j)
   # random.shuffle(a)
   # random.shuffle(b)
   # np.save(save_dir_matFiles+'a.npy', a)
   # np.save(save_dir_matFiles+'b.npy', b)

    # a = np.load('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/a.npy')
    # b = np.load('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/b.npy')
    #
    # for k in range(boundary.shape[0]):
    #     t = 0
    #     for i in a:
    #         for j in b:
    #             if (authentic_edges[k,i,j] !=0) and (t <= counts_pos[k]):
    #                 t = t+1
    #                 neg3 = third_hop_attri[k,i,j]
    #                 negative3.append(neg3)
    #                 neg2 = second_hop_attri[k,i,j]
    #                 negative2.append(neg2)
    #                 neg1 = one_hop_attri[k,i,j]
    #                 negative1.append(neg1)
    #
    #
    # negative3 = np.array(negative3)
    # negative2 = np.array(negative2)
    # negative1 = np.array(negative1)
    #
    # cPickle.dump(negative3, open(save_dir_matFiles+'neg_3_100.pkl', 'wb'))
    # cPickle.dump(negative2, open(save_dir_matFiles+'neg_2_100.pkl', 'wb'))
    # cPickle.dump(negative1, open(save_dir_matFiles+'neg_1_100.pkl', 'wb'))


    #%%
        
    print("--------TEST PROCESS--------")   
    
#     # read test images
# #    test_filenames = np.random.permutation(os.listdir(test_dir + img_dir)).tolist()
# #    with open(save_dir_matFiles + 'filenames_Columbia_test10.pkl', 'wb') as fid:
# #        cPickle.dump(filenames, fid)
#
    # with open(save_dir_matFiles + 'filenames_colum40.pkl', 'rb') as fid:
    #    test_filenames = cPickle.load(fid)

    # ext_test_imgs = boundary_extension(All_train_test+img_dir, test_filenames)
    # print("test extended imgs shape:", ext_test_imgs.shape)
    #
    # test_GC = read_gc(All_train_test+GC_dir, test_filenames)
    # print("test GC shape:", test_GC.shape)
    #
    # test_All_edge = read_all_edge(All_train_test+All_edge_dir, test_filenames)
    # print("test All edge shape:", test_All_edge.shape)
    #
    # # attributes
    # pca_params_1 = cPickle.load(open(save_dir_matFiles+'saab_params_1_100.sav', 'rb'))
    # pca_params_2 = cPickle.load(open(save_dir_matFiles+'saab_params_2_100.sav', 'rb'))
    # pca_params_3 = cPickle.load(open(save_dir_matFiles+'saab_params_3_100.sav', 'rb'))
    #
    # test_one_hop_attri = test_one_hop(ext_test_imgs, pca_params_1)
    #
    # h5f = h5py.File(save_dir_matFiles+'test_1_100.h5', 'w')
    # h5f.create_dataset('attribute', data=test_one_hop_attri)
    # h5f.close()
    #
    # # h5f = h5py.File(save_dir_matFiles+'test_1_100.h5', 'r')
    # # test_one_hop_attri = h5f['attribute'][:]
    #
    # test_sec_hop_attri = test_second_hop(test_one_hop_attri, pca_params_2)
    #
    # h5f = h5py.File(save_dir_matFiles+'test_2_100.h5', 'w')
    # h5f.create_dataset('attribute', data=test_sec_hop_attri)
    # h5f.close()
    #
    # # h5f = h5py.File(save_dir_matFiles+'test_2_100.h5', 'r')
    # # test_sec_hop_attri = h5f['attribute'][:]
    #
    # # pooled_test_sec_hop_attri = pooling(test_sec_hop_attri)
    # #
    # test_third_hop_attri = test_third_hop(test_sec_hop_attri, pca_params_3)
    # #
    # # test_third_hop_attri = np.zeros((test_GC.shape[0], test_GC.shape[1], test_GC.shape[2], test_third_hop_attri_reduced.shape[-1]))
    # # for k in range(test_third_hop_attri_reduced.shape[0]):
    # #     test_third_hop_attri[k] = cv2.resize(test_third_hop_attri_reduced[k], (test_GC.shape[2], test_GC.shape[1]),
    # #                                     interpolation=cv2.INTER_LINEAR)
    # #
    # h5f = h5py.File(save_dir_matFiles+'test_3_100.h5', 'w')
    # h5f.create_dataset('attribute', data=test_third_hop_attri)
    # h5f.close()

#%%
    #
    # kernel1 = np.ones((3, 3), np.uint8)
    # kernel2 = np.ones((10,10), np.uint8)
    #
    # num_cores = int(multiprocessing.cpu_count() / 2)
    # n_samp = test_GC.shape[0]
    #
    # test_spliced_authentic = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(select_samples)(test_GC, test_All_edge, kernel1, kernel2, k) for k in range(n_samp))
    #
    # print(len(test_spliced_authentic))
    # test_spliced_authentic = np.array(test_spliced_authentic)
    # print(test_spliced_authentic.shape)
    # test_boundary = test_spliced_authentic[:, 0, :, :]
    # test_authentic_edges = test_spliced_authentic[:, 1, :, :]
    #
    # print("spliced mask shape:", test_boundary.shape)
    # print("authentic mask shape:", test_authentic_edges.shape)


#%%

    # test_positive3 = []
    # test_negative3 = []
    # test_positive2 = []
    # test_negative2 = []
    # test_positive1 = []
    # test_negative1 = []
    #
    # # test_pos_count = [[] for i in range(test_GC.shape[0])]
    # # test_neg_count = [[] for i in range(test_GC.shape[0])]
    # # test_pos_mask = np.zeros((test_GC.shape))
    # # test_neg_mask = np.zeros((test_GC.shape))
    # #
    # #
    # for k in range(test_boundary.shape[0]):
    #     for i in range(test_boundary.shape[1]):
    #         for j in range(test_boundary.shape[2]):
    #             if test_All_edge[k,i,j] != 0 and test_boundary[k,i,j] != 0:   # spliced boundary pixel
    #
    #                 # test_pos_mask[k,i,j] = 1
    #                 # test_pos_count[k].append([i,j,0])
    #
    #                 test_pos3 = test_third_hop_attri[k,i,j]
    #                 test_positive3.append(test_pos3)
    #                 test_pos2 = test_sec_hop_attri[k,i,j]
    #                 test_positive2.append(test_pos2)
    #                 test_pos1 = test_one_hop_attri[k,i,j]
    #                 test_positive1.append(test_pos1)
    #
    #
    #             if test_All_edge[k,i,j] != 0 and test_authentic_edges[k,i,j] != 0:   # authentic boundary pixel
    #
    #                 # test_neg_mask[k,i,j] = 1
    #                 # test_neg_count[k].append([i,j,0])
    #
    #                 test_neg3 = test_third_hop_attri[k,i,j]
    #                 test_negative3.append(test_neg3)
    #                 test_neg2 = test_sec_hop_attri[k,i,j]
    #                 test_negative2.append(test_neg2)
    #                 test_neg1 = test_one_hop_attri[k,i,j]
    #                 test_negative1.append(test_neg1)
    #
    #
    # test_positive3 = np.array(test_positive3)
    # test_negative3 = np.array(test_negative3)
    #
    # test_positive2 = np.array(test_positive2)
    # test_negative2 = np.array(test_negative2)
    #
    # test_positive1 = np.array(test_positive1)
    # test_negative1 = np.array(test_negative1)
    #
    # np.save(save_dir_matFiles+'test_pos_3_100.npy', test_positive3)
    # np.save(save_dir_matFiles+'test_neg_3_100.npy', test_negative3)
    #
    # np.save(save_dir_matFiles+'test_pos_2_100.npy', test_positive2)
    # np.save(save_dir_matFiles+'test_neg_2_100.npy', test_negative2)
    #
    # np.save(save_dir_matFiles+'test_pos_1_100.npy', test_positive1)
    # np.save(save_dir_matFiles+'test_neg_1_100.npy', test_negative1)


# %%
    # LAG on training samples

    positive1 = np.load(save_dir_matFiles + 'pos_1_100.npy')
    positive2 = np.load(save_dir_matFiles + 'pos_2_100.npy')
    positive3 = np.load(save_dir_matFiles + 'pos_3_100.npy')

    negative1 = cPickle.load(open(save_dir_matFiles + 'neg_1_100.pkl', 'rb'))
    negative2 = cPickle.load(open(save_dir_matFiles + 'neg_2_100.pkl', 'rb'))
    negative3 = cPickle.load(open(save_dir_matFiles + 'neg_3_100.pkl', 'rb'))

    print("spliced and natural number:", positive1.shape[0], negative1.shape[0])
    print("------ START LAG UNIT ------")

    num_clusters = [20, 20]
    SAVE = {}
    class_list = [1, 0]

    pos_label = np.ones((positive1.shape[0],), dtype=int)
    neg_label = np.zeros((negative1.shape[0],), dtype=int)
    train_labels = np.concatenate((pos_label, neg_label), axis=0)

    if classifier == 1:
        train_feature = np.concatenate((positive1, negative1), axis=0)

    elif classifier == 2:
        train_feature = np.concatenate((positive2, negative2), axis=0)

    elif classifier == 3:
        train_feature = np.concatenate((positive3, negative3), axis=0)

    elif classifier == 4:
        positive_ensem = np.concatenate((positive1, positive2, positive3), axis=1)
        negative_ensem = np.concatenate((negative1, negative2, negative3), axis=1)
        train_feature = np.concatenate((positive_ensem, negative_ensem), axis=0)

    train_feature_reduce = LAG_Unit(train_feature, train_labels=train_labels, class_list=class_list,
                                    SAVE=SAVE, num_clusters=num_clusters, alpha=20, Train=True)

    print(train_feature_reduce.shape)
    print("------ LAG UNIT FINISHED ------")


#%%
    # LAG on test samples

    test_positive1 = np.load(save_dir_matFiles + 'test_pos_1_100.npy')
    test_negative1 = np.load(save_dir_matFiles + 'test_neg_1_100.npy')

    test_positive2 = np.load(save_dir_matFiles + 'test_pos_2_100.npy')
    test_negative2 = np.load(save_dir_matFiles + 'test_neg_2_100.npy')

    test_positive3 = np.load(save_dir_matFiles + 'test_pos_3_100.npy')
    test_negative3 = np.load(save_dir_matFiles + 'test_neg_3_100.npy')

    test_pos_labels = np.ones((test_positive1.shape[0],))
    test_neg_labels = np.zeros((test_negative1.shape[0],))
    test_labels = np.concatenate((test_pos_labels, test_neg_labels), axis=0)

    if classifier == 1:
        test_feature = np.concatenate((test_positive1, test_negative1), axis=0)

    elif classifier == 2:
        test_feature = np.concatenate((test_positive2, test_negative2), axis=0)

    elif classifier == 3:
        test_feature = np.concatenate((test_positive3, test_negative3), axis=0)

    elif classifier == 4:
        test_positive_ensem = np.concatenate((test_positive1, test_positive2, test_positive3), axis=1)
        test_negative_ensem = np.concatenate((test_negative1, test_negative2, test_negative3), axis=1)
        test_feature = np.concatenate((test_positive_ensem, test_negative_ensem), axis=0)

    test_feature_reduce = LAG_Unit(test_feature, train_labels=None, class_list=None, SAVE=SAVE, num_clusters=None, alpha=None, Train=False)
    print(test_feature_reduce.shape)

# %%

    print("-------TRAINING CLASSIFIERS-------")
    RF = True
    min_samples_split = 2
    max_features = 'auto'
    bootstrap = True
    max_depth = 20
    min_samples_leaf = 2
    n_estimators_rf = 100


    clf1 = RandomForestClassifier(n_jobs=int(multiprocessing.cpu_count() / 2), verbose=0, class_weight='balanced',
                                      n_estimators=n_estimators_rf, min_samples_split=min_samples_split,
                                      max_features=max_features, bootstrap=bootstrap,
                                      max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    clf1.fit(train_feature_reduce, train_labels)

    predicted_train_label = clf1.predict(train_feature_reduce)

    C_train = confusion_matrix(train_labels, predicted_train_label)

    per_class_accuracy_train = np.diag(C_train.astype(np.float32)) / np.sum(C_train.astype(np.float32), axis=1)
    print(per_class_accuracy_train)

#%%    

    test_predicted_label = clf1.predict(test_feature_reduce)
    C_test = confusion_matrix(test_labels, test_predicted_label)
    per_class_accuracy_test = np.diag(C_test.astype(np.float32)) / np.sum(C_test.astype(np.float32), axis=1)
    print(per_class_accuracy_test)

    # test_output = output_map(test_predicted_label, test_pos_count, test_neg_count, test_positive1)
