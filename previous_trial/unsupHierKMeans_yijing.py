# v2020.05.20
# unsupervised HierKmean
'''
# zigzag scanning to split training groupds
# terminatation: too few total samples/ reach to target leaf numbers/ reach to target mse
# leaf nodes are labeled from 0 to K as prediction
'''

import numpy as np
import copy
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

# LBG initialization
def Init_LBG(X, sep_num=2):
    c1 = np.mean(X, axis=0).reshape(1,-1)
    st = np.std(X, axis=0).reshape(1,-1)
    c2 = c1 + st
    new_centroids = np.concatenate((c1, c2), axis=0)
    return new_centroids  


def zigzag(n):
    '''zigzag rows'''
    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)
    xs = range(n)
    idx = {index: n for n, index in enumerate(sorted(
        ((x, y) for x in xs for y in xs),
        key=compare
    ))}
    zigzag_block = np.zeros((n,n))
    k = idx.copy()
    key = k.keys()
    for k in key:
        zigzag_block[k] = idx[k]
    return zigzag_block

 
 

class tKMeans():
    def __init__(self, MSE=10, min_percent=0.05, leaf_node=10, standerdization=0, group_block=4):
        self.data = {}
        self.targetMSE = MSE
        self.targetNode = leaf_node
        self.min_sample_num = -1
        self.min_sample_percent = min_percent
        self.trained = False
        self.pred= []
        self.leaf_num = 0
        self.standerdization = standerdization
        self.group_block = group_block # 4x4 --> 16 groups of samples
        
    def calMSE_(self, nX):
        nMSE = sklearn.metrics.pairwise.euclidean_distances(nX, np.mean(nX, axis=0).reshape(1, -1))
        return nMSE
    
    def split_group(self):
        groups = []
        indicator_block = zigzag(self.group_block)
        groups = np.tile(indicator_block,(int(np.ceil(self.h/np.float(self.group_block))),int(np.ceil(self.w/np.float(self.group_block)))))
        groups = groups[:self.h,:self.w].reshape(1,self.h,self.w)
        group_label = np.tile(groups,(self.n,1,1)).reshape(-1)
        return group_label.astype('int')
        
        
    
    def check_split_(self):
        k = self.data.copy()
        key = k.keys()
        for k in key:
            
            if k == 'KMeans':
                continue
            
#            if self.data[k]['splittable']==1 
            if self.data[k]['splitted']==0:
                if np.mean(self.calMSE_(self.data[k]['X'])) > self.targetMSE:
                    
                    # train kmean on selected samples
                    init_centroids = Init_LBG(self.data[k]['X'][self.data[k]['train_idx']])
                    kmeans = KMeans(n_clusters=2, init=init_centroids).fit(self.data[k]['X'][self.data[k]['train_idx']])
                    
                    # predict kmean on all the samples
                    pred = kmeans.predict(self.data[k]['X'])
                    
                    for i in range(2):
                        self.data[k+str(i)] = {'X':self.data[k]['X'][pred == i],
                                              'group': self.data[k]['group'][pred==i],
                                              'splitted':0,
                                              'use_groups':self.data[k]['use_groups'],
                                              'leaf':0}
                        
                        if np.sum(pred==i)<self.min_sample_num:
                            # it will stop splitting, too few total samples
                            self.data[k+str(i)]['splitted'] = 1
#                            self.data[k+str(i)]['splittable']=0
                            self.data[k+str(i)]['num'] = self.data[k+str(i)]['X'].shape[0]
                            self.data[k+str(i)]['X'] = np.array([1])
                            self.data[k+str(i)]['leaf'] = 1
                            self.leaf_num+=1
                        else:
#                            self.data[k+str(i)]['splittable']=1
                            self.data[k+str(i)]['train_idx'] = np.where(self.data[k+str(i)]['group']<self.data[k+str(i)]['use_groups'])[0]
                            
#                            if len(self.data[k+str(i)]['train_idx'])<self.min_sample_num:
                            while len(self.data[k+str(i)]['train_idx'])<self.min_sample_num:
                                # too few training samples --> add in one more group
                                self.data[k+str(i)]['use_groups'] += 1
                                self.data[k+str(i)]['train_idx'] = np.where(self.data[k+str(i)]['group']<self.data[k+str(i)]['use_groups'])[0]
                    
                    self.data[k]['X'] = np.array([1])
                    self.data[k]['KMeans'] = kmeans
                    self.data[k]['splitted'] = 1
                    
                else:
                    self.data[k]['splitted'] = 1
#                    self.data[k]['splittable'] = 0
                    self.data[k]['num'] = self.data[k]['X'].shape[0]
                    self.data[k]['X'] = np.array([1])
                    self.data[k]['leaf'] = 1
                    self.leaf_num+=1
                    
                
    def fit(self, X):
        print(X.shape)
        self.n, self.h, self.w, self.c = X.shape
        X = X.reshape(-1,self.c)
        
        if self.standerdization:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
            
        self.min_sample_num = X.shape[0] * self.min_sample_percent
        
        group_labels = self.split_group()
        
        init_centroids = Init_LBG(X[group_labels < 1])
        kmeans = KMeans(n_clusters=2, init=init_centroids).fit(X[group_labels < 1])
        pred = kmeans.predict(X)
        
        self.data['KMeans'] = kmeans
        
        for i in range(2):
            self.data[str(i)] = {'X':X[pred == i], 'group': group_labels[pred==i], 'splitted':0, 'splittable':1, 'use_groups':1,'leaf':0}
            self.data[str(i)]['train_idx'] = np.where(self.data[str(i)]['group'] < self.data[str(i)]['use_groups'])[0]
            
        while self.leaf_num<self.targetNode:
            self.check_split_()
    
        k = self.data.copy()
        key = k.keys()
        
        k_idx = 0   # ??
        for k in key:
            if k == 'KMeans':
                continue
            if self.data[k]['leaf']==1:
                self.data[k]['k_idx'] = np.copy(k_idx)
                k_idx +=1
            else:
                if self.data[k]['splitted']==0: 
                    # mark the last round new leaves
                    self.data[k]['num'] = self.data[k]['X'].shape[0]
                    self.data[k]['X'] = np.array([1])
                    self.data[k]['leaf']=1
                    self.leaf_num+=1
                    self.data[k]['k_idx'] = np.copy(k_idx)
                    k_idx +=1
        self.trained = True

    def predict(self, X):
        print('>>>>>>>>>>>>>>>>>>>>>>>Clustering')
        X = X.reshape(-1,self.c)
        if self.standerdization:
            X = self.scaler.transform(X)
        pred = np.zeros(X.shape[0])
        k = self.data.copy()
        key = k.keys()
        self.depth = 1
        for k in key:
            if k != 'KMeans':
                if len(k)>self.depth:
                    self.depth = len(k)
        del k
        print('>>>>>>>>>>>>>>>>>>>>>>>TREE depth = {}'.format(self.depth))
        tmp_data = []
        label = self.data['KMeans'].predict(X)
        
        for k in range(2):
            tmp_data.append({'X':X[label==k], 'idx':np.arange(X.shape[0])[label==k], 'id':str(k)})
            
        for i in range(self.depth):
            tmp = []
            for j in range(len(tmp_data)):
                if tmp_data[j]['X'].shape[0] == 0:
                    continue
                if self.data[tmp_data[j]['id']]['leaf']==0:
                    label = self.data[tmp_data[j]['id']]['KMeans'].predict(tmp_data[j]['X'])
                    for k in range(2):
                        tmp.append({'X':tmp_data[j]['X'][label==k], 'idx':tmp_data[j]['idx'][label==k], 'id':tmp_data[j]['id']+str(k)})
                else:
                    pred[tmp_data[j]['idx']] = np.copy(self.data[tmp_data[j]['id']]['k_idx'])
            tmp_data = tmp
            
        return pred
    
            
#################################### Test #####################################
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import h5py
    import pickle
    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/multi_resolution/features/'
    h5f = h5py.File(save_dir_matFiles + 'original_resolution/context/train_context.h5', 'r')
    context = h5f['attribute'][:]
    print(context.shape)

    print(" > This is a test example: ")
    X = context

    # X = tt[np.newaxis,:,:,:].astype('float64')
    print(" input feature shape: %s"%str(X.shape))
    
    hier = tKMeans(MSE=0.001, min_percent=0.01, leaf_node=16, standerdization=1, group_block=4)
    hier.fit(X)
    pred = hier.predict(X)
    
    
    print(pred.shape)
    print("------- DONE -------\n")


    # analysis of splitting result of unsupervised Hierarchical Kmeans on context
    pickle.load(open(save_dir_matFiles + 'original_resolution/context/neg_target.pkl', 'rb'))


    pickle.load(open(save_dir_matFiles + 'original_resolution/context/neg_pixel_context.pkl', 'rb'))
