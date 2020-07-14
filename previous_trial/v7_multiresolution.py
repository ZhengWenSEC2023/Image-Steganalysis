#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:19:51 2019

@author: Yao Zhu
"""

import numpy as np
import _pickle as cPickle
from skimage import io
import time
from joblib import Parallel, delayed
import multiprocessing
import cv2
import h5py

from itertools import product
from scipy.ndimage import gaussian_filter

from framework.tree_PixelHop import PixelHop_Unit, PixelHop_fit



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

def boundary_extension(image_dir, filenames, augment=None):
    # image_dir = train_dir+img_dir
    
    n_samp = len(filenames)
    filenames_filtered = [filenames[i] for i in range(n_samp)]
    extended_imgs = []
    for i in range(n_samp):
        img_samp = io.imread(image_dir + filenames_filtered[i])
        # img_samp_lab = color.rgb2lab(img_samp)
        if augment == None:
            img_rotate = img_samp
        else:
            img_flip = cv2.flip(img_samp, 1)
            img_rotate = cv2.rotate(img_flip, augment)


        extended_imgs.append(img_rotate)
        
    extended_imgs = np.array(extended_imgs)
    
    return extended_imgs

def read_gc(edges_dir, filenames, augment):
    
    n_samp = len(filenames)
    filenames_filtered = [filenames[i] for i in range(n_samp)]
    gc = []
    for i in range(n_samp):
        edge_samp = io.imread(edges_dir + filenames_filtered[i][:-4]+'_gc.png')
        if augment == None:
            edge_rotate = edge_samp
        else:
            edge_flip = cv2.flip(edge_samp, 1)
            edge_rotate = cv2.rotate(edge_flip, augment)

        gc.append(edge_rotate)

    gc = np.array(gc)
    
    return gc

def read_all_edge(edges_dir, filenames, augment):
    
    n_samp = len(filenames)
    filenames_filtered = [filenames[i] for i in range(n_samp)]
    alledge = []
    for i in range(n_samp):
        alledge_samp = io.imread(edges_dir + filenames_filtered[i])

        if augment == None:
            alledge_rotate = alledge_samp
        else:
            alledge_flip = cv2.flip(alledge_samp, 1)
            alledge_rotate = cv2.rotate(alledge_flip, augment)

        alledge.append(alledge_rotate)

    alledge = np.array(alledge)
    
    return alledge
        




def select_samples(GCs, All_edges, kernel1, kernel2, k):

    GC = GCs[k]  # 256, 384
    All_edge = All_edges[k]  # 256, 384

    Splicing_edge_image = RobertsAlogrithm(GC)  # very thin
    Splicing_edge_image[Splicing_edge_image != 0] = 255

    # for i in range(GC.shape[0]):
    #     for j in range(GC.shape[1]):
    #         if Splicing_edge_image[i][j] != 0:
    #             Splicing_edge_image[i][j] = 255

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
    n_samp = attribute.shape[0]

    pooled_attri = np.zeros((n_samp, int(attribute.shape[1]/2), int(attribute.shape[2]/2), attribute.shape[3]))
    for i in range(0, attribute.shape[1], 2):
       for j in range(0, attribute.shape[2], 2):
           pooled_attri[:,int(i/2),int(j/2),:] = attribute[:,i,j,:]
    # pooled_attri = block_reduce(attribute, block_size=(1,2,2,1), func=np.max)


    return pooled_attri


def fit_pca_shape(datasets,depth):
    factor=np.power(4,depth)
    length_x=int(datasets.shape[-2]/factor)
    length_y = int(datasets.shape[-1]/factor)

    idx1=range(0,length_x,4)
    idx2=[i+4 for i in idx1]
    idy1 = range(0,length_y, 4)
    idy2 = [j+4 for j in idy1]

    data_lattice=[datasets[:,i:j,k:l] for ((i,j),(k,l)) in product(zip(idx1,idx2),zip(idy1,idy2))]
    data_lattice=np.array(data_lattice)
    print('fit_pca_shape: data_lattice.shape: {}'.format(data_lattice.shape))

#     #shape reshape
#     data=np.reshape(data_lattice,(data_lattice.shape[0]*data_lattice.shape[1],data_lattice.shape[2],4,4))
#     print('fit_pca_shape: reshape: {}'.format(data.shape))
    return data_lattice

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def one_hop(extended_imgs):

    # # 8 nearest neighbors
    # a = time.time()
    # n_samp = extended_imgs.shape[0]
    #
    # num_cores = int(multiprocessing.cpu_count()/2)
    #
    # one_hop_attri_saab = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(concatenate)(extended_imgs, k, 2, False) for k in range(n_samp))
    # print(type(one_hop_attri_saab))
    #
    # one_hop_attri_saab = np.array(one_hop_attri_saab)
    # print(one_hop_attri_saab.shape)
    #
    # b = time.time()
    # print("time for concatenate 1st :", b-a)

    # # Saab dimension reduction from 27 dim to 12 dim
    # pca_params_1 = Saab_Getkernel(one_hop_attri_saab, "27")
    # c = time.time()
    # print("time for kernel:", c-b)
    #
    # one_hop_attri = Saab_Getfeature(one_hop_attri_saab, pca_params_1)
    # d = time.time()
    # print("time for feature:", d-c)

    one_hop_attri_saab = PixelHop_Unit(extended_imgs, getK= True, idx_list= None, dilate=1, window_size=4, pad='reflect', weight_root=save_dir_matFiles+'weight',
                                       Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None, PCA_ener_percent=0.99,
                                       useDC=False, stride=None, getcov=0, split_spec=0, hopidx=1)

    print("Hop-1 feature ori shape:", one_hop_attri_saab.shape)
    one_hop_attri_response = PixelHop_fit(weight_name = save_dir_matFiles+'weight', feature_ori = one_hop_attri_saab, split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)

    print("one hop attribute shape:", one_hop_attri_response.shape)



    return one_hop_attri_response


def second_hop(one_hop_attri_response, idx_stop_list):

    # a = time.time()
    # num_cores = int(multiprocessing.cpu_count() / 2)
    # n_samp = one_hop_attri.shape[0]
    #
    # sec_hop_attri_saab = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(concatenate)(one_hop_attri, k, 3, False) for k in range(n_samp))
    # print(type(sec_hop_attri_saab))
    #
    # sec_hop_attri_saab = np.array(sec_hop_attri_saab)
    # print(sec_hop_attri_saab.shape)
    #
    # b = time.time()
    # print("time for concatenate 2nd :", b - a)
    #
    # # Saab dimension reduction from 27 dim to 12 dim
    # pca_params_2 = Saab_Getkernel(sec_hop_attri_saab, "107")
    # c = time.time()
    # print("time for kernel:", c-b)
    #
    # sec_hop_attri = Saab_Getfeature(sec_hop_attri_saab, pca_params_2)
    # d = time.time()
    # print("time for feature:", d-b)
    #
    # print("second hop attribute shape:", sec_hop_attri.shape)

    all_nodes = np.arange(one_hop_attri_response.shape[-1]).tolist()
    for i in idx_stop_list:
        if i in all_nodes:
            all_nodes.remove(i)
    intermediate_idx = all_nodes

    sec_hop_attri_saab = PixelHop_Unit(one_hop_attri_response, dilate=7, window_size=1, idx_list=intermediate_idx, getK=True, pad='reflect',
                                       weight_root=save_dir_matFiles + 'weight',
                                       Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None, PCA_ener_percent=0.95,
                                       useDC=False, stride=None, getcov=0, split_spec=1, hopidx=2)

    print("Hop-2 feature ori shape:", sec_hop_attri_saab.shape)

    sec_hop_attri_response = PixelHop_fit(weight_name=save_dir_matFiles + 'weight', feature_ori=sec_hop_attri_saab,
                                        split_spec=1, hopidx=2, Pass_Ener_thrs=0.1)

    print("sec hop attri shape:", sec_hop_attri_response.shape)

    return sec_hop_attri_response




#%%
def test_one_hop(ext_test_imgs):

    # a = time.time()
    # n_samp = ext_test_imgs.shape[0]
    # num_cores = int(multiprocessing.cpu_count() / 2)
    #
    # test_one_hop_attri_saab = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(concatenate)(ext_test_imgs, k, 2, True) for k in range(n_samp))
    # print(type(test_one_hop_attri_saab))
    # test_one_hop_attri_saab = np.array(test_one_hop_attri_saab)
    # print(test_one_hop_attri_saab.shape)
    # b = time.time()
    # print("time for concatenate 1st test:", b - a)
    #
    # # Saab dimension reduction from 27 dim to 12 dim
    # test_one_hop_attri = Saab_Getfeature(test_one_hop_attri_saab, pca_params_1)
    # print("Test 1st hop attribute shape:", test_one_hop_attri.shape)

    # gt = np.zeros((ext_test_imgs.shape[0], ext_test_imgs.shape[1], ext_test_imgs.shape[2]))
    # test_one_hop_attri = PixelHop_Unit(ext_test_imgs, gt, dilate=np.array([1]), num_AC_kernels=27, pad='reflect',
    #                               weight_root=save_dir_matFiles + 'weight',
    #                               getK=True, Pass_Ener_thrs=0.2, energy_percent=0.02, useDC=False, stride=None,
    #                               getcov=0, split_spec=1, hopidx=1)

    test_one_hop_attri_saab = PixelHop_Unit(ext_test_imgs, dilate=1, window_size=4, idx_list=None, getK=False, pad='reflect',
                                       weight_root=save_dir_matFiles + 'weight',
                                            Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None, PCA_ener_percent=0.99,
                                            useDC=False, stride=None, getcov=0, split_spec=0, hopidx=1)


    test_one_hop_attri_biased = PixelHop_fit(weight_name=save_dir_matFiles + 'weight',feature_ori=test_one_hop_attri_saab,
                                             split_spec=0, hopidx=1, Pass_Ener_thrs=0.1)

    return test_one_hop_attri_biased

def test_second_hop(test_one_hop_attri_response, idx_stop_list):

    # a = time.time()
    # n_samp = test_one_hop_attri.shape[0]
    # num_cores = int(multiprocessing.cpu_count() / 2)
    #
    # test_sec_hop_attri_saab = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(concatenate)(test_one_hop_attri, k, 3, False) for k in range(n_samp))
    # print(type(test_sec_hop_attri_saab))
    # test_sec_hop_attri_saab = np.array(test_sec_hop_attri_saab)
    # print(test_sec_hop_attri_saab.shape)
    # b = time.time()
    # print("time for concatenate 2nd :", b - a)
    #
    # # Saab dimension reduction from 27 dim to 12 dim
    # test_sec_hop_attri = Saab_Getfeature(test_sec_hop_attri_saab, pca_params_2)
    # print("Test 2nd hop attribute shape:", test_sec_hop_attri.shape)

    all_nodes = np.arange(test_one_hop_attri_response.shape[-1]).tolist()
    for i in idx_stop_list:
        if i in all_nodes:
            all_nodes.remove(i)
    intermediate_idx = all_nodes

    test_sec_hop_attri_saab = PixelHop_Unit(test_one_hop_attri_response, dilate=7, window_size=1, idx_list=intermediate_idx, getK=False, pad='reflect',
                                            weight_root=save_dir_matFiles + 'weight',
                                            Pass_Ener_thrs=0.1, Leaf_Ener_thrs=0.0001, num_kernels=None, PCA_ener_percent=0.95,
                                            useDC=False, stride=None, getcov=0, split_spec=1, hopidx=2)

    print("sec hop neighborhood construction shape:", test_sec_hop_attri_saab.shape)

    test_sec_hop_attri_response = PixelHop_fit(weight_name=save_dir_matFiles + 'weight', feature_ori=test_sec_hop_attri_saab,
                                        split_spec=1, hopidx=2, Pass_Ener_thrs=0.1)

    print("sec hop attri shape:", test_sec_hop_attri_response.shape)
    
    return test_sec_hop_attri_response
    





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
    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/multi_resolution/features/'



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
    # with open('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/' + 'filenames_columbia180.pkl', 'wb') as fid:
    #    cPickle.dump(filenames_colum, fid)

    with open('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/' + 'filenames_alltrain100.pkl', 'rb') as fid:
        train_filenames = cPickle.load(fid)

    with open('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/' + 'filenames_columbia180.pkl', 'rb') as fid:
        test_filenames = cPickle.load(fid)

    train_filenames = train_filenames
    test_filenames = test_filenames

    # delete
    for i in range(len(train_filenames)):
        img_name = train_filenames[i]
        if img_name in test_filenames:
            idx = test_filenames.index(img_name)
            test_filenames.pop(idx)





    # train_filenames.pop(4)


    ext_train_imgs = boundary_extension(All_train_test+img_dir, train_filenames, augment=cv2.ROTATE_90_COUNTERCLOCKWISE)
    # ext_train_imgs = block_reduce(ext_train_imgs, block_size=(1,4,4,1), func=np.mean)
    # ext_train_imgs = ext_train_imgs[:,:,:,0]  # only L channel from LAB color space
    # ext_train_imgs = np.expand_dims(ext_train_imgs, axis=-1)

    print("extended images shape:", ext_train_imgs.shape)

    All_edges = read_all_edge(All_train_test+All_edge_dir, train_filenames, augment=cv2.ROTATE_90_COUNTERCLOCKWISE)
    # All_edges = block_reduce(All_edges, block_size=(1,2,2), func=np.mean)
    # All_edges[All_edges > 0.5] = 1
    # All_edges[All_edges < 0.5] = 0

    print("All edges shape:", All_edges.shape)

    GC = read_gc(All_train_test+GC_dir, train_filenames, augment=cv2.ROTATE_90_COUNTERCLOCKWISE)
    # GC = block_reduce(GC, block_size=(1,2,2), func=np.mean)
    # GC[GC > 0.5] = 1
    # GC[GC < 0.5] = 0
    print("GC shape:", GC.shape)

    # for k in range(3):
    #     cv2.imwrite(save_dir_matFiles + 'gaussian_edge/' + train_filenames[k][:-4] + 'alledge_%d.png'% k, All_edges[k]*255)


    # del filenames_casia, filenames_casia_used, filenames_colum, filenames_colum_used
    # del ext_train_imgs_casia, ext_train_imgs_colum, All_edges_casia, All_edges_colum, GC_casia, GC_colum







    # %%
    start = time.time()
    kernel1 = np.ones((5, 5), np.uint8)     # width of spliced boundary
    kernel2 = np.ones((10, 10), np.uint8)

    num_cores = int(multiprocessing.cpu_count() / 2)
    n_samp = GC.shape[0]

    spliced_authentic = Parallel(n_jobs=num_cores, backend='multiprocessing')(
        delayed(select_samples)(GC, All_edges, kernel1, kernel2, k) for k in range(n_samp))

    print(len(spliced_authentic))
    spliced_authentic = np.array(spliced_authentic)
    print(spliced_authentic.shape)
    boundary = spliced_authentic[:, 0, :, :]    # spliced boundary pixels
    authentic_edges = spliced_authentic[:, 1, :, :]

    print("spliced mask shape:", boundary.shape)
    print("authentic mask shape:", authentic_edges.shape)

    end = time.time()
    print("time for selecting:", end - start)


    # for k in range(boundary.shape[0]):
    #     count_spliced = 0
    #     for i in range(boundary.shape[1]):
    #         for j in range(boundary.shape[2]):
    #             if boundary[k,i,j] != 0 and All_edges[k,i,j] !=0:
    #                 count_spliced = count_spliced + 1
    #     if count_spliced == 0:
    #         train_filenames[k] = '0'

    # train_filenames = np.array(train_filenames)
    # train_filenames = train_filenames[train_filenames != '0']
    # print(len(train_filenames))
    #
    # with open(save_dir_matFiles + 'filenames_alltrain100.pkl', 'wb') as fid:
    #    cPickle.dump(train_filenames, fid)






    #%% modify binary label to be a range

    altered_spliced = np.zeros((boundary.shape))
    altered_authentic = np.zeros((authentic_edges.shape))
    # change label value for spliced edge and authentic edge
    for k in range(boundary.shape[0]):
        for i in range(boundary.shape[1]):
            for j in range(boundary.shape[2]):
                if boundary[k,i,j] != 0 and All_edges[k,i,j]!= 0:   # spliced boundary pixel
                    altered_spliced[k,i,j] = 1

                if authentic_edges[k,i,j] != 0:
                    altered_authentic[k,i,j] = -1

    filtered_spliced = np.zeros((altered_spliced.shape))
    filtered_authentic = np.zeros((altered_authentic.shape))
    for k in range(altered_spliced.shape[0]):
        filtered_spliced[k] = gaussian_filter(altered_spliced[k], sigma=1)
        filtered_authentic[k] = gaussian_filter(altered_authentic[k], sigma=1)


#%% normalize modified range to 0~1, -1~0



    # # visualize convoluted spliced boundary
    # vis_spliced_edge = np.zeros((boundary.shape))
    # vis_authentic_edge = np.zeros((authentic_edges.shape))
    #
    # for k in range(boundary.shape[0]):
    #     for i in range(boundary.shape[1]):
    #         for j in range(boundary.shape[2]):
    #             if filtered_img_alledge[k][i][j] > 0:
    #                 vis_spliced_edge[k][i][j] = filtered_img_alledge[k][i][j]
    #             if filtered_img_alledge[k][i][j] <= 0:
    #                 vis_authentic_edge[k][i][j] = filtered_img_alledge[k][i][j]


    # use this below

    # make range from 0~1, -1~0
    filtered_spliced_01 = np.zeros((filtered_spliced.shape))
    filtered_authentic_01 = np.zeros((filtered_authentic.shape))

    for k in range(filtered_spliced.shape[0]):
        filtered_spliced_01[k] = (filtered_spliced[k] - np.min(filtered_spliced[k]))/(np.max(filtered_spliced[k]) - np.min(filtered_spliced[k]))     # 0-1
        filtered_authentic_01[k] = (filtered_authentic[k] - np.min(filtered_authentic[k]))/(np.max(filtered_authentic[k]) - np.min(filtered_authentic[k])) - 1     # -1-0


    print(np.max(filtered_spliced_01[0]), np.min(filtered_spliced_01[0]))
    print(np.max(filtered_authentic_01[0]), np.min(filtered_authentic_01[0]))


    # use this above






    # for k in range(3):
    #     plt.figure()
    #     plt.imshow(altered_spliced[k]+altered_authentic[k], cmap='coolwarm')
    #     plt.colorbar()
    #     plt.savefig(save_dir_matFiles + train_filenames[k][:-4] + '_gt_%d.png' % k)
    #
    #     plt.figure()
    #     plt.imshow(filtered_authentic_01[k] + filtered_spliced_01[k], cmap='coolwarm')
    #     plt.colorbar()
    #     plt.axis('off')
    #     plt.savefig(save_dir_matFiles + 'original_resolution/augment/' + train_filenames[k][:-4] + 'gaussian_edge_%d.png' % k)


    # use below

    # name_loc_prob_train = []
    # for k in range(boundary.shape[0]):
    #     dic = {}
    #     dic['train_name'] = train_filenames[k]
    #     dic['spliced_loc'] = []
    #     dic['authentic_loc'] = []
    #     name_loc_prob_train.append(dic)


    # %%

    print(">>    FEATURE EXTRACTION     <<")
    start = time.time()
    one_hop_attri_response = one_hop(ext_train_imgs)  # window size 5*5*3
    print(one_hop_attri_response.shape)

    h5f = h5py.File(save_dir_matFiles+'original_resolution/augment/train_1_1-1_9by9_flip_counterclock90.h5', 'w')
    h5f.create_dataset('attribute', data=one_hop_attri_response)
    h5f.close()

    # h5f = h5py.File(save_dir_matFiles + 'original_resolution/augment/train_1_1-1_9by9.h5', 'r')
    # one_hop_attri_response = h5f['attribute'][:]
    # print(one_hop_attri_response.shape)

    end = time.time()
    print("Time 1:", end - start)

    # idx_stop_list = np.arange(one_hop_attri_response.shape[-1]//2, one_hop_attri_response.shape[-1])
    #
    # one_hop_attri = one_hop_attri_response[:,:,:, idx_stop_list]
    # print("Hop-1 leaf node attri shape:", one_hop_attri.shape)
    #
    #
    # second_hop_attri_response = second_hop(one_hop_attri_response, idx_stop_list=idx_stop_list)
    #
    # h5f = h5py.File(save_dir_matFiles + 'original_resolution/augment/train_2_1-4_5by5.h5', 'w')
    # h5f.create_dataset('attribute', data=second_hop_attri_response)
    # h5f.close()
    #
    # # h5f = h5py.File(save_dir_matFiles+'train_2_100.h5', 'r')
    # # second_hop_attri_response = h5f['attribute'][:]
    # # print(second_hop_attri_response.shape)
    #

    # resize to 1/2, 1/4
    one_hop_attri = one_hop_attri_response
    # one_hop_attri = block_reduce(one_hop_attri_response, block_size=(1,2,2,1), func=np.mean)
    # second_hop_attri = second_hop_attri_response

    # end2 = time.time()
    # print("Time 2:", end2-end)

#%%
    print(">>    SELECTING PIXEL FEATURES    <<")

    print(one_hop_attri.shape)
    # print(second_hop_attri.shape)
    # print(third_hop_attri.shape)
    pos_target = []
    neg_target = []
    # positive3 = []
    # negative3 = []
    # positive2 = []
    # negative2 = []
    positive1 = []
    negative1 = []
    counts_pos = np.zeros((GC.shape[0],), dtype=int)

    for k in range(boundary.shape[0]):
        count = 0
        for i in range(boundary.shape[1]):
            for j in range(boundary.shape[2]):
                if filtered_spliced_01[k,i,j] > 0 and altered_spliced[k,i,j] != 0 :   # spliced boundary pixel
                    count = count + 1

                    target = filtered_spliced_01[k,i,j]
                    pos_target.append(target)
                    # pos3 = third_hop_attri[k,i,j]
                    # positive3.append(pos3)
                    # pos2 = second_hop_attri[k,i,j]
                    # positive2.append(pos2)
                    pos1 = one_hop_attri[k,i,j]
                    positive1.append(pos1)

                    # name_loc_prob_train[k]['spliced_loc'].append([i, j, target])

        # if count == 0:
        #     name = train_filenames[k]
        #     cv2.imwrite(save_dir_matFiles + 'pos0sample/%s' % name, boundary[k])
        #     cv2.imwrite(save_dir_matFiles + 'pos0sample/%s_gc.png' % name[:-4], GC[k]*255)
        #     cv2.imwrite(save_dir_matFiles + 'pos0sample/%s_alledge.png' % name[:-4], All_edges[k]*255)
        counts_pos[k] = count

    print("total number of positive pixels:", counts_pos)
    print("pos target range:", np.max(pos_target), np.min(pos_target))
    np.save(save_dir_matFiles + 'original_resolution/augment/pos_target_flip_counterclock90.npy', pos_target)


    # positive3 = np.array(positive3)
    # positive2 = np.array(positive2)
    positive1 = np.array(positive1)
    # np.save(save_dir_matFiles+'pos_3_100_pool.npy', positive3)
    # np.save(save_dir_matFiles+'original_resolution/augment/pos_2_100.npy', positive2)
    np.save(save_dir_matFiles+'original_resolution/augment/pos_1_100_flip_counterclock90.npy', positive1)

#%%
    # a = []
    # b = []
    # for i in range(boundary.shape[1]):
    #     a.append(i)
    # for j in range(boundary.shape[2]):
    #     b.append(j)
    # random.shuffle(a)
    # random.shuffle(b)
    # np.save('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/a_8.npy', a)
    # np.save('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/b_8.npy', b)

    # a = np.load('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/a.npy')
    # b = np.load('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/b.npy')
    # print(a.shape, b.shape)

    for k in range(boundary.shape[0]):
        t = 0
        for i in range(256):
            for j in range(384):
                if (filtered_authentic_01[k,i,j] < 0) and altered_authentic[k,i,j] != 0 :
                    t = t+1
                    target = filtered_authentic_01[k,i,j]
                    neg_target.append(target)
                    # neg3 = third_hop_attri[k,i,j]
                    # negative3.append(neg3)
                    # neg2 = second_hop_attri[k,i,j]
                    # negative2.append(neg2)

                    neg1 = one_hop_attri[k,i,j]
                    negative1.append(neg1)

                    # name_loc_prob_train[k]['authentic_loc'].append([i, j, target])


    # negative3 = np.array(negative3)
    # negative2 = np.array(negative2)
    negative1 = np.array(negative1)

    neg_target = np.array(neg_target)
    print("neg target range:", np.max(neg_target), np.min(neg_target))
    cPickle.dump(neg_target, open(save_dir_matFiles + 'original_resolution/clear_test/neg_target_all.pkl', 'wb'))

    # cPickle.dump(negative3, open(save_dir_matFiles+'neg_3_100_pool.pkl', 'wb'))
    # cPickle.dump(negative2, open(save_dir_matFiles+'original_resolution/clear_test/neg_2_100.pkl', 'wb'))
    cPickle.dump(negative1, open(save_dir_matFiles+'original_resolution/clear_test/neg_1_100_all.pkl', 'wb'))



    # %%

    # with open(save_dir_matFiles + 'name_loc_prob_train_100.pkl', 'wb') as fid:
    #     cPickle.dump(name_loc_prob_train, fid)


#%%   test process extract features
    print("--------TEST PROCESS--------")


#     # read test images
# #    test_filenames = np.random.permutation(os.listdir(test_dir + img_dir)).tolist()
# #    with open(save_dir_matFiles + 'filenames_Columbia_test10.pkl', 'wb') as fid:
# #        cPickle.dump(filenames, fid)
#
    # with open(save_dir_matFiles + 'filenames_colum40.pkl', 'rb') as fid:
    #    test_filenames = cPickle.load(fid)

    ext_test_imgs = boundary_extension(All_train_test+img_dir, test_filenames)
    # ext_test_imgs = block_reduce(ext_test_imgs, block_size=(1,2,2,1), func=np.mean)
    # ext_test_imgs = ext_test_imgs[:,:,:,0]
    # ext_test_imgs = np.expand_dims(ext_test_imgs, axis=-1)
    print("test extended imgs shape:", ext_test_imgs.shape)


    test_GC = read_gc(All_train_test+GC_dir, test_filenames)
    # test_GC = block_reduce(test_GC, block_size=(1,2,2), func=np.mean)
    # test_GC[test_GC > 0.5] = 1
    # test_GC[test_GC < 0.5] = 0
    print("test GC shape:", test_GC.shape)

    test_All_edge = read_all_edge(All_train_test+All_edge_dir, test_filenames)
    # test_All_edge = block_reduce(test_All_edge, block_size=(1,2,2), func=np.mean)
    # test_All_edge[test_All_edge > 0.5] = 1
    # test_All_edge[test_All_edge < 0.5] = 0
    print("test All edge shape:", test_All_edge.shape)



    #%%

    print(">>    FEARURE EXTRATION    <<")
    # attributes

    test_one_hop_attri_response = test_one_hop(ext_test_imgs)
    print(test_one_hop_attri_response.shape)

    # test_one_hop_attri = test_one_hop_attri_response[:,:,:,idx_stop_list]


    h5f = h5py.File(save_dir_matFiles+'original_resolution/clear_test/test_1_1-1_9by9.h5', 'w')
    h5f.create_dataset('attribute', data=test_one_hop_attri_response)
    h5f.close()

    # h5f = h5py.File(save_dir_matFiles+'original_resolution/ssl/test_1_1-1_9by9.h5', 'r')
    # test_one_hop_attri_response = h5f['attribute'][:]
    # print("test_hop_1_attri shape:", test_one_hop_attri_response.shape)

    # test_sec_hop_attri_response = test_second_hop(test_one_hop_attri_response, idx_stop_list)
    #
    # h5f = h5py.File(save_dir_matFiles+'original_resolution/clear_test/test_2.h5', 'w')
    # h5f.create_dataset('attribute', data=test_sec_hop_attri_response)
    # h5f.close()



    test_one_hop_attri = test_one_hop_attri_response
    # test_one_hop_attri = block_reduce(test_one_hop_attri_response, block_size=(1,2,2,1), func=np.mean)
    # test_sec_hop_attri = test_sec_hop_attri_response
    print(test_one_hop_attri.shape)


    # h5f = h5py.File(save_dir_matFiles+'test_2_10.h5', 'r')
    # test_sec_hop_attri = h5f['attribute'][:]
    # print("test_hop_2_attri shape:", test_sec_hop_attri.shape)

    # pooled_test_sec_hop_attri = pooling(test_sec_hop_attri)
    #
    # test_third_hop_attri_reduced = test_third_hop(pooled_test_sec_hop_attri, pca_params_3)
    #
    # test_third_hop_attri = np.zeros((GC.shape[0], GC.shape[1], GC.shape[2], third_hop_attri_reduced.shape[-1]))
    # for k in range(test_third_hop_attri_reduced.shape[0]):
    #     test_third_hop_attri[k] = cv2.resize(test_third_hop_attri_reduced[k], (test_GC.shape[2], test_GC.shape[1]),
    #                                     interpolation=cv2.INTER_LINEAR)
    #
    # h5f = h5py.File(save_dir_matFiles+'test_3_100_pool.h5', 'w')
    # h5f.create_dataset('attribute', data=test_third_hop_attri)
    # h5f.close()


#%% test spliced boundary and authentic boundary selection

    kernel1 = np.ones((5,5), np.uint8)
    kernel2 = np.ones((10,10), np.uint8)

    num_cores = int(multiprocessing.cpu_count() / 2)
    n_samp = test_GC.shape[0]

    test_spliced_authentic = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(select_samples)(test_GC, test_All_edge, kernel1, kernel2, k) for k in range(n_samp))

    print(len(test_spliced_authentic))
    test_spliced_authentic = np.array(test_spliced_authentic)
    print(test_spliced_authentic.shape)
    test_boundary = test_spliced_authentic[:, 0, :, :]
    test_authentic_edges = test_spliced_authentic[:, 1, :, :]

    print("spliced mask shape:", test_boundary.shape)
    print("authentic mask shape:", test_authentic_edges.shape)


#%%   save positive & negative pixel location in dict
    name_loc_prob = []
    for k in range(test_boundary.shape[0]):
        dic = {}
        dic['test_name'] = test_filenames[k]
        name_loc_prob.append(dic)



    for k in range(test_boundary.shape[0]):
        name_loc_prob[k]['spliced_loc'] = []
        name_loc_prob[k]['authentic_loc'] = []
        for i in range(test_boundary.shape[1]):
            for j in range(test_boundary.shape[2]):
                if test_All_edge[k,i,j] != 0 and test_boundary[k,i,j] != 0:   # spliced boundary pixel
                    name_loc_prob[k]['spliced_loc'].append([i,j])

                if test_All_edge[k,i,j] != 0 and test_authentic_edges[k,i,j] != 0:  # authentic edge pixel
                    name_loc_prob[k]['authentic_loc'].append([i,j])



    with open(save_dir_matFiles + 'original_resolution/clear_test/name_loc.pkl', 'wb') as fid:
        cPickle.dump(name_loc_prob, fid)






#%%  select positive & negative pixels of test imgs

    # test_positive3 = []
    # test_negative3 = []
    # test_positive2 = []
    # test_negative2 = []
    test_positive1 = []
    test_negative1 = []
    # test_positive1_pkl = [[] for i in range(test_boundary.shape[0])]
    # test_negative1_pkl = [[] for i in range(test_boundary.shape[0])]


    for k in range(test_boundary.shape[0]):
        for i in range(test_boundary.shape[1]):
            for j in range(test_boundary.shape[2]):
                if test_All_edge[k,i,j] != 0 and test_boundary[k,i,j] != 0:   # spliced boundary pixel


                    # test_pos3 = test_third_hop_attri[k,i,j]
                    # test_positive3.append(test_pos3)
                    # test_pos2 = test_sec_hop_attri[k,i,j]
                    # test_positive2.append(test_pos2)
                    test_pos1 = test_one_hop_attri[k,i,j]
                    test_positive1.append(test_pos1)
                    # test_positive1_pkl[k].append(test_pos1)


                if test_All_edge[k,i,j] != 0 and test_authentic_edges[k,i,j] != 0:   # authentic boundary pixel

                    # test_neg3 = test_third_hop_attri[k,i,j]
                    # test_negative3.append(test_neg3)
                    # test_neg2 = test_sec_hop_attri[k,i,j]
                    # test_negative2.append(test_neg2)
                    test_neg1 = test_one_hop_attri[k,i,j]
                    test_negative1.append(test_neg1)
                    # test_negative1_pkl[k].append(test_neg1)

        # test_positive1_pkl[k] = np.array(test_positive1_pkl[k])
        # test_negative1_pkl[k] = np.array(test_negative1_pkl[k])


    # test_positive3 = np.array(test_positive3)
    # test_negative3 = np.array(test_negative3)

    # test_positive2 = np.array(test_positive2)
    # test_negative2 = np.array(test_negative2)

    test_positive1 = np.array(test_positive1)
    test_negative1 = np.array(test_negative1)
    #
    # np.save(save_dir_matFiles+'test_pos_3_100_pool.npy', test_positive3)
    # np.save(save_dir_matFiles+'test_neg_3_100_pool.npy', test_negative3)

    # np.save(save_dir_matFiles+'original_resolution/clear_test/test_pos_2.npy', test_positive2)
    # np.save(save_dir_matFiles+'original_resolution/clear_test/test_neg_2.npy', test_negative2)

    np.save(save_dir_matFiles+'original_resolution/clear_test/test_pos_1.npy', test_positive1)
    np.save(save_dir_matFiles+'original_resolution/clear_test/test_neg_1.npy', test_negative1)
    # cPickle.dump(test_positive1_pkl, open(save_dir_matFiles+'original_resolution/clear_test/test_pos_1.pkl', 'wb'))
    # cPickle.dump(test_negative1_pkl, open(save_dir_matFiles+'original_resolution/clear_test/test_neg_1.pkl', 'wb'))


# %%   !!!!!!IGNORE THE REST!!!!!!!
    print(test_positive1.shape,test_negative1.shape)
    # print(test_positive2.shape, test_negative2.shape)
    print(positive1.shape, negative1.shape)
    # print(positive2.shape, negative2.shape)

