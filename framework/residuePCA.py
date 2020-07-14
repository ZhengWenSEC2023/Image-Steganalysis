#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:03:22 2020

@author: yyj-linux
"""

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from numpy import linalg as LA
from skimage.measure import block_reduce
import pickle
import time
import matplotlib.pyplot as plt
from framework.saab import Saab



class ResPCA():
    def __init__(self, weight_name, kernel_sizes, target_ener_percent=0.5, useDC=False, getcov=0, split_spec=1):
        self.weight_name = weight_name
        self.kernel_sizes = kernel_sizes
        # self.useDC = useDC
        self.target_ener_percent = target_ener_percent
        self.getcov = getcov
        self.split_spec=split_spec
        self.tree = []
        self.leaf_num = 0
        self.energy = []
    
# =============================================================================
#     def splice_RP(self, feature, useDC):
#         saab = Saab(self.kernel_sizes, num_kernels=1, getcov=self.getcov, useDC=useDC)
#         transformed = saab.fit_transform(feature)
#         pca_params = saab.pca_params
#         res = feature - np.matmul(transformed, pca_params['kernel']) - pca_params['feature_expectation']
#         # res = feature - np.matmul(transformed, pca_params['kernel']) 
#         # plt.imshow(pca_params['kernel'].reshape(5,5),cmap='gray')
#         # plt.savefig('kernel'+str(time.time())+'.png')
#         # plt.show()
#         return res, pca_params   
#         
# =============================================================================
    def build_resTree(self, pixelhop_feature):   
        saab = Saab(self.kernel_sizes, energy_percent = 1-self.target_ener_percent,getcov=self.getcov, useDC=1)
        saab.fit(pixelhop_feature)
        ener_curve = saab.energy
        self.tree = saab.pca_params
        self.leaf_num = saab.num_kernels
        self.energy = ener_curve.reshape(-1,1)
#        next_hop
        return ener_curve
    
    def fit(self, pixelhop_feature):
        print("------------------- Start: Fit - Build Residue Tree")
        t0 = time.time()
        ener_curve = self.build_resTree(pixelhop_feature)
        print("plot")
        plt.figure(0)
        plt.plot(ener_curve,'bo-')
        plt.xticks(range(len(ener_curve)))
        plt.savefig(self.weight_name[:-4]+'.png')
        plt.close(0)   
        # print("save")
        # fw = open(self.weight_name, 'wb')
        # pickle.dump(resTree, fw)
        # fw.close()
        print("       <Info>        Save Residue Tree as name: %s"%str(self.weight_name))
        print("------------------- End: Fit -> using %10f seconds"%(time.time()-t0))           
    
    def transform(self, test_feature, train=0):
        print("------------------- Start: Transform - Traverse the Residue Tree")
        t0 = time.time()

        response_for_next_hop = np.matmul(test_feature - self.tree['feature_expectation'],np.transpose(self.tree['kernel']))
        response_for_next_hop = response_for_next_hop + 1 / np.sqrt(response_for_next_hop.shape[3]) * self.tree['bias']
        print("       <Info>        response shape: {}".format(response_for_next_hop.shape))
        print("------------------- End: Transform -> using %10f seconds"%(time.time()-t0))      
        return response_for_next_hop
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        