#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:19:51 2019

@author: mac
"""

import numpy as np
import _pickle as cPickle
from skimage import io
from sklearn.decomposition import PCA
import time
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
from sklearn.metrics import matthews_corrcoef as mcc


#%%

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
        

#def Saab_Getkernel(data, train_labels, kernel_sizes, num_kernels):
#    # read data
#    # train_images, train_labels, test_images, test_labels, class_list = data.import_data(FLAGS.use_classes)
#    train_images = data
#    
##    if flag == 'pos':
##        train_labels = np.ones((train_images.shape[0],), dtype=int)
##        
##    else:
##        train_labels = np.zeros((train_images.shape[0],), dtype=int)
##    test_labels = np.load('new_data_result/test_label.npy')
#    class_list = [0,1]
#
#    print('       START SAAB TRAINING KERNEL       ')
#    print('       Training image size:', train_images.shape)
##    print('Testing_image size:', test_images.shape)
#    
#    kernel_sizes=saab.parse_list_string(kernel_sizes)
#    if num_kernels:
#        num_kernels=saab.parse_list_string(num_kernels)
#    else:
#        num_kernels=None
#    energy_percent=None
#    use_num_images=-1
#    print('       Parameters:')
#    print('       use_classes:', class_list)
#    print('       Kernel_sizes:', kernel_sizes)
#    print('       Number_kernels:', num_kernels)
#    print('       Energy_percent:', energy_percent)
#    print('       Number_use_images:', use_num_images)
#
#    pca_params=saab.multi_Saab_transform(train_images, train_labels,
#                         kernel_sizes=kernel_sizes,
#                         num_kernels=num_kernels,
#                         energy_percent=energy_percent,
#                         use_num_images=use_num_images,
#                         use_classes=class_list)
##    # save data
##    fw=open('Saab_output/' + neg_filter_test10.pkl','wb')
##    pickle.dump(pca_params, fw)    
##    fw.close()
#    return pca_params
#
#
#def Saab_Getfeature(data, filters):
#    
#    train_images = data
#    
#    pca_params = filters
#
#    print('       Training image size:', train_images.shape)
##    print('Testing_image size:', test_images.shape)
#
#    
#    # Training
#    print('       Training feature extraction--------')
#    feature=saab.initialize(train_images, pca_params) 
#    print("       S4 shape:", feature.shape)
#    print('       Finish Feature Extraction subnet--------')
#    
#    return feature[:,0,0,:]

def one_hop(extended_imgs, label):
    
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
    
    # Saab dimension reduction from 27 dim to 12 dim
#    pca_params = Saab_Getkernel(one_hop_attri_saab, GC, kernel_size, num_kernels)
#    one_hop_attri = Saab_Getfeature(one_hop_attri_saab, pca_params)
    
    # pca dimension reduction
    reshaped_data = one_hop_attri_saab.reshape((-1, 27))
#    reshaped_label = label.reshape((-1))
    
    pca1 = PCA(n_components = 12)
    reshaped_data_new = pca1.fit_transform(reshaped_data)
    one_hop_attri = reshaped_data_new.reshape((100, 256, 384, 12))
    
    print("one hop attribute shape:", one_hop_attri.shape)
  
    
    return pca1, one_hop_attri
#%%
def second_hop(one_hop_attri):
    
    # 8 nearest neighbors
    n_samp = one_hop_attri.shape[0]
    h = one_hop_attri.shape[1] #256
    w = one_hop_attri.shape[2] #384
    
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
                
    # dimension reduction
    reshaped_sec_hop_attri_saab = sec_hop_attri_saab.reshape((-1, 108))
    
    pca2 = PCA(n_components=50)
    reshaped_sec_hop_attri_saab_new = pca2.fit_transform(reshaped_sec_hop_attri_saab)
    
    sec_hop_attri = reshaped_sec_hop_attri_saab_new.reshape((n_samp, h,w,50))
                    
    print("second hop attribute shape:", sec_hop_attri.shape)
    
    return pca2, sec_hop_attri

#%%
def third_hop(sec_hop_attri):
    
    # 8 nearest neighbors
    n_samp = sec_hop_attri.shape[0]
    h = sec_hop_attri.shape[1] #256
    w = sec_hop_attri.shape[2] #384
    
    #sec_hop_attri shape: 100, 256, 384, 50
    padded_sec_hop_attri = np.lib.pad(sec_hop_attri, ((0,0), (4,4), (4,4), (0,0)), 'symmetric')   
    print("padded attribute shape", padded_sec_hop_attri.shape) # 100, 264, 392, 50
    
    third_hop_attri_saab = np.zeros((n_samp, h, w, 450))
    
    for k in range(n_samp):
        for i in range(4,4+h): #4-260
            for j in range(4,4+w): #4-388
                atttotal = []
                for m in range(50):
                    top = np.concatenate((padded_sec_hop_attri[k,i-4,j-4,m], padded_sec_hop_attri[k,i-4,j,m], padded_sec_hop_attri[k,i-4,j+4,m]), axis=None)
                    mid = np.concatenate((padded_sec_hop_attri[k,i,j-4,m], padded_sec_hop_attri[k,i,j,m], padded_sec_hop_attri[k,i,j+4,m]), axis=None)
                    bot = np.concatenate((padded_sec_hop_attri[k,i+4,j-4,m], padded_sec_hop_attri[k,i+4,j,m], padded_sec_hop_attri[k,i+4,j+4,m]), axis=None)
                    attslice = np.concatenate((top, mid, bot), axis=None) #9
                    atttotal.extend(attslice)
                atttotal = np.array(atttotal)
                third_hop_attri_saab[k,i-4,j-4,:] = atttotal
                
    # dimension reduction
    reshaped_third_hop_attri_saab = third_hop_attri_saab.reshape((-1, 450))
    
    pca3 = PCA(n_components=150)
    reshaped_third_hop_attri_saab_new = pca3.fit_transform(reshaped_third_hop_attri_saab)
    
    third_hop_attri = reshaped_third_hop_attri_saab_new.reshape((n_samp, h,w,150))
                    
    print("third hop attribute shape:", third_hop_attri.shape)
    
    return pca3, third_hop_attri

#%%
def test_one_hop(ext_test_imgs, pca1):
    
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
    
    reshaped_test_one_hop_long = test_one_hop_long.reshape((-1, 27))
    test_one_hop_attri_flatten = pca1.fit_transform(reshaped_test_one_hop_long)
    
    test_one_hop_attri = test_one_hop_attri_flatten.reshape((n_samp, h, w, 12))
    print("Test images one hop attribute shape:",test_one_hop_attri.shape) # 10, 256, 384, 12
    
    return test_one_hop_attri

def test_second_hop(test_one_hop_attri, pca2):
    
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
                
    # dimension reduction
    reshaped_test_sec_hop_long = test_sec_hop_long.reshape((-1, 108))   
    test_sec_hop_attri_flatten = pca2.fit_transform(reshaped_test_sec_hop_long)
    test_sec_hop_attri = test_sec_hop_attri_flatten.reshape((n_samp, h,w, 50))
    print("Test images second hop attribute shape:",test_sec_hop_attri.shape) # 10, 256, 384, 50
    
    return test_sec_hop_attri
    

#%%
if __name__=='__main__':
    
    start = time.time()
    train_dir = "/Users/mac/Desktop/mcl_2019/image_splicing/example_files/CASIA2/"
    test_dir = '/Users/mac/Desktop/mcl_2019/image_splicing/example_files/COLUMBIA/'
    img_dir = 'img/'
    GC_dir = 'GC_ver0and1/'
    All_edge_dir = 'All_edges_0p2_ver0and1/'
    save_dir = '/Users/mac/Desktop/mcl_2019/image_splicing/Output_Text_Files/'
    save_dir_matFiles = '/Users/mac/Desktop/mcl_2019/image_splicing/Output_Mat_Files/'
    n_estimators_rf = 100
    classifier = 1 # use which hop attribute

#%%    
    # save train images name into a pickle
#    filenames = np.random.permutation(os.listdir(train_dir + img_dir)).tolist()
#    with open(save_dir_matFiles + 'filenames_CASIA2_test100.pkl', 'wb') as fid:
#        cPickle.dump(filenames, fid)
        
#     read train images name from pickle    
    with open(save_dir_matFiles + 'filenames_CASIA2_test100.pkl', 'rb') as fid:
        filenames = cPickle.load(fid)

    ext_train_imgs = boundary_extension(train_dir+img_dir, filenames)
#    All_edges = read_all_edge(train_dir+All_edge_dir, filenames)
    GC = read_gc(train_dir+GC_dir, filenames)
    
    # create train_labels for training pixels: GC can be used as label
    kernel_size = "1"
    num_kernels = "12"
    
    pca1, one_hop_attri = one_hop(ext_train_imgs, GC)
    # cPickle.dump(pca1, open(save_dir_matFiles+'pca_hop1.sav', 'wb'))
    # cPickle.dump(one_hop_attri, open(save_dir_matFiles+'first_hop_attributes.pkl', 'wb'))
    
    pca2, second_hop_attri = second_hop(one_hop_attri)
    
    cPickle.dump(pca2, open(save_dir_matFiles+'pca_hop2.sav', 'wb'))
    # cPickle.dump(one_hop_attri, open(save_dir_matFiles+'first_hop_attributes.pkl', 'wb'))
    
    with open(save_dir_matFiles + 'second_hop_attributes.pkl', 'rb') as fid:
        second_hop_attri = cPickle.load(fid)
    
    pca3, third_hop_attri = third_hop(second_hop_attri)
    cPickle.dump(pca3, open(save_dir_matFiles+'pca_hop3.sav', 'wb'))

#%%    
    # train classifiers random forest classifier
    # label: GC, data: third hop attribute
    print("-------TRAINING CLASSIFIERS-------")
#    second_hop_attri = cPickle.load(open(save_dir_matFiles+'second_hop_attributes.pkl', 'rb'))
    
    clf1 = RandomForestClassifier(n_jobs=int(multiprocessing.cpu_count()/2), verbose=0, class_weight='balanced',n_estimators=n_estimators_rf)
    label_flatten = GC.reshape((-1))
    
    if classifier == 1: # use first hop attribute only
        one_hop_attri_flatten = one_hop_attri.reshape((-1, 12))
        
        clf1.fit(one_hop_attri_flatten, label_flatten)
        predicted_train_label = clf1.predict(one_hop_attri_flatten)
        mcc_score = mcc(label_flatten, predicted_train_label)
        
    elif classifier == 2: # use second hop attributes only
        sec_hop_flatten = second_hop_attri.reshape((-1, 50))
        
        clf1.fit(sec_hop_flatten, label_flatten)
        predicted_train_label = clf1.predict(sec_hop_flatten)
        mcc_score = mcc(label_flatten, predicted_train_label)
        
    elif classifier == 3: # use third hop attribute only
        
        third_hop_flatten = third_hop_attri.reshape((-1, 150))
        
        clf1.fit(third_hop_flatten, label_flatten)
        predicted_train_label = clf1.predict(third_hop_flatten)
        mcc_score = mcc(label_flatten, predicted_train_label)
        
    print("Training MCC score:", mcc_score)    
        
    print("--------TEST PROCESS--------")   
    
#    # read test images
#    test_filenames = np.random.permutation(os.listdir(test_dir + img_dir)).tolist()
#    with open(save_dir_matFiles + 'filenames_Columbia_test10.pkl', 'wb') as fid:
#        cPickle.dump(filenames, fid)
#        
##    with open(save_dir_matFiles + 'filenames_Columbia_test10.pkl', 'rb') as fid:
##        test_filenames = cPickle.load(fid)
#        
#    ext_test_imgs = boundary_extension(test_dir+img_dir, test_filenames)
#    test_GC = read_gc(test_dir+GC_dir, test_filenames)
#    test_label_flatten = test_GC.reshape((-1))
#    
#    # attributes
#    test_one_hop_attri = test_one_hop(ext_test_imgs, pca1)
#    
#    test_sec_hop_attri = test_second_hop(test_one_hop_attri, pca2)
 
    # classify by trained classifiers
    if classifier == 1:
        test_one_hop_flatten = test_one_hop_attri.reshape((-1, test_one_hop_attri.shape[-1]))
        predicted_test_label = clf1.predict(test_one_hop_flatten)
        
        test_mcc_score = mcc(test_label_flatten, predicted_test_label)
        
        print("Test MCC score using 1 hop:", test_mcc_score)
        
    elif classifier == 2:
        test_sec_hop_flatten = test_sec_hop_attri.reshape((-1, test_sec_hop_attri.shape[-1]))
        predicted_test_label = clf1.predict(test_sec_hop_flatten)
        
        test_mcc_score = mcc(test_label_flatten, predicted_test_label)
        
        print("TestMCC score using 2 hop:", test_mcc_score)
        
    
    

    

    

    end = time.time()
    print("time consumed:", end-start)
    