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

def window_process(samples, kernel_size, stride):
    '''
    Create patches
    :param samples: [num_samples, feature_height, feature_width, feature_channel]
    :param kernel_size: int i.e. patch size
    :param stride: int
    :return patches: flattened, [num_samples, output_h, output_w, feature_channel*kernel_size^2]

    '''
    samples = np.pad(samples,((0,0),(int(kernel_size/2),int(kernel_size/2)),(int(kernel_size/2),int(kernel_size/2)),(0,0)),'reflect')
    n, h, w, c = samples.shape
    output_h = (h - kernel_size) // stride + 1
    output_w = (w - kernel_size) // stride + 1
    patches = view_as_windows(samples, (1, kernel_size, kernel_size, c), step=(1, stride, stride, c))
    patches = patches.reshape(n, output_h, output_w, kernel_size * kernel_size, c)
    return patches


def sample_indicator(gt):
#    indicator = np.empty(gt.shape)
#    indicator[gt==1]=1
#    posNum = np.sum(gt==1)
#    for i in range(gt.shape[0]):
#        idX, idY = np.where(gt[i]==0)
#        randIdx = np.random.permutation(idX.size)[:int(posNum/gt.shape[0]+1)]
#        idX = idX[randIdx]
#        idY = idY[randIdx]
#        for m in range(idX.size):
#            indicator[i,idX[m],idY[m]] = 1
#    indicator = indicator.astype('int64')   
    thrs=0.001
    indicator = np.zeros(gt.shape)
    indicator[gt>=thrs]=1
    posNum = np.sum(gt>=thrs)
    for i in range(gt.shape[0]):
        idX, idY = np.where(gt[i]<thrs)
        randIdx = np.random.permutation(idX.size)[:int(posNum/gt.shape[0]+1)]
        idX = idX[randIdx]
        idY = idY[randIdx]
        for m in range(idX.size):
            indicator[i,idX[m],idY[m]] = 1
    indicator = indicator.astype('int64')   
    return indicator


def PixelHop_batchSample(feature, indicator, dilate, pad):
    for i in range(int(feature.shape[0]/20)):
        temp = PixelHop_Neighbour(feature[i*20:(i+1)*20,:,:,:], dilate, pad)
        temp = temp.reshape(-1,temp.shape[-2],temp.shape[-1])
        if i==0:
            feature_pool = np.empty((np.sum(indicator),temp.shape[-2], temp.shape[-1]))
            START = 0
        END = np.sum(indicator[:(i+1)*20])
        feature_pool[START:END] = np.copy(temp[indicator[i*20:(i+1)*20].reshape(-1)==1])
        del temp
        START = np.copy(END)
#    feature_pool = feature_pool.reshape(feature_pool.shape[0]*feature_pool.shape[1], -1)
    print(">>>>>>>>>>>>>>>>>>> finish batch sampling")    
    return feature_pool
    
    
def PixelHop_Neighbour(feature, dilate, window_size, pad):
    # feature: input feature, shape: n, x, y, channel (3/1)
    #  dilate: dilate number, only when window size=3, dilate
    #  window_size: 5 or 3, if == 5, no dilation, if == 3, use dilated window
    print("------------------- Start: PixelHop_Neighbour")
    print("   <Neighborhood Info>Input feature shape: %s"%str(feature.shape))
    print("   <Neighborhood Info>.   dilate: %s" % str(dilate))
    print("   <Neighborhood Info>window size: %s"%str(window_size))
    print("   <Neighborhood Info>filter size:", str(2*window_size+1), str(2*window_size+1), str(feature.shape[-1]))
    #print("   <Info>padding: %s"%str(pad))
    #  t0 = time.time()
    S = feature.shape

    if pad == 'reflect':
        feature = np.pad(feature, ((0,0),(window_size*dilate, window_size*dilate),(window_size*dilate, window_size*dilate),(0,0)), 'reflect')
    elif pad == 'zeros':
        feature = np.pad(feature, ((0,0),(window_size*dilate, window_size*dilate),(window_size*dilate, window_size*dilate),(0,0)), 'constant', constant_values=0)

    print("   <Neighborhood Info> padded image shape:", feature.shape)

    # dilate = np.array(dilate).astype('int64')

    res = np.zeros((S[1], S[2], S[0], (2*window_size+1)**2 , S[3]))


    feature = np.moveaxis(feature, 0, 2)# 256, 384, 100, 3


    idx = (np.arange(2*window_size+1) - window_size) * dilate
    for i in range(window_size*dilate, feature.shape[0] - window_size*dilate):
        for j in range(window_size*dilate, feature.shape[1] - window_size*dilate):
            tmp = []
            for ii in idx:
                for jj in idx:
                    if ii == 0 and jj == 0:
                        continue
                    iii = i+ii
                    jjj = j+jj
                    tmp.append(feature[iii, jjj])
            tmp.append(feature[i,j])
            tmp = np.array(tmp)   # 9, 100, 3
            tmp = np.moveaxis(tmp,0,1)   # 100, 9, 3
            res[i-window_size*dilate, j-window_size*dilate] = tmp   # 256, 384, 100, 9, 3

    res = np.moveaxis(res, 2, 0) # 100, 256, 384, 9, 3
    print("   <Neighborhood Info>Output feature shape: %s"%str(res.shape))
    #print("------------------- End: PixelHop_Neighbour -> using %10f seconds"%(time.time()-t0))
    return res

def cal_entropy(X, MIN=0, MAX=1,num_bins=100,leafidx=0, hopidx=0):
    H=0.0
    STEP = np.float(MAX-MIN)/num_bins
    bb,_,_ = plt.hist(X,bins=np.arange(MIN,MAX+STEP,STEP))
    plt.figure(0)
    plt.hist(X,bins=np.arange(MIN,MAX+STEP,STEP))
    plt.savefig('./plots/hist_hop'+str(hopidx)+'_leaf'+str(leafidx)+'.png')
    plt.close(0)
#    plt.show()
    prob = np.divide(bb,np.sum(bb))
    prob[prob<1e-5]=1
    for i in range(prob.size):
        H += -prob[i]*(math.log(prob[i])/math.log(prob.size))
    return H

def decide_pass_idx1(feature, leaf_energy_last, Leaf_Ener_thrs, Pass_Ener_thrs, hopidx=1):
    # using variance
    var = np.zeros(feature.shape[-1])
    for cc in range(feature.shape[-1]):
        var[cc] = np.var(feature[:,:,:,cc].reshape(-1))    # every padded channel variance
    sorted_var = -1*np.sort(-1*var)
    plt.figure(0)
    plt.plot(sorted_var,'bo-')
    plt.ylabel('variance')
    plt.xlabel('leaf idx')
    plt.title('sorted variance of channels in hop '+str(hopidx-1))
    plt.savefig('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/plots/variance_of_hop'+str(hopidx-1)+'_sorted.png')
    plt.close(0)
    logmap1 = leaf_energy_last>Leaf_Ener_thrs    # same as not_end_node
    logmap2 = var>(Pass_Ener_thrs*np.max(sorted_var))
    # logmap2 = var>sorted_var[int(Pass_Ener_thrs*len(sorted_var))]
    print('{}*{}'.format(Pass_Ener_thrs,len(sorted_var)))
    print('cutting var = {}'.format(sorted_var[int(Pass_Ener_thrs*len(sorted_var))]))
    idx_list = np.where(logmap1==1)[0].tolist()
    return idx_list


def PixelHop_Unit(feature, getK=True, idx_list = None, dilate=0, window_size = 2, pad='reflect', weight_root='./tmp',
                  Pass_Ener_thrs=0.4,  Leaf_Ener_thrs=0.001, num_kernels = None, PCA_ener_percent=None,
                  useDC=True, stride = None, getcov=0, split_spec=1, hopidx=1):
    '''
    Pass_Ener_thrs: the threshold to select the channel for passing down
    PCA_ener_percent: energy for PCA
    Leaf_Ener_thrs: the threshold for the leaf termination energy
    '''
    print("=========== Create weight root ...")    
    if not os.path.isdir(weight_root+str(hopidx)+'/'):os.makedirs(weight_root+str(hopidx)+'/')    
    print("=========== Start: PixelHop_Unit")
    t0 = time.time()
    if getK == True:
        print(">>>>>====== Training ======<<<<<")
        # indicator = sample_indicator(gt)
        total_leaf_num = 0.0
        all_subTree = {}
        leaf_energy = []

        if split_spec == 1 :        
            if hopidx>1:# start from 2hop
                # fr = open(weight_root+str(hopidx-1)+'/leaf_energy.pkl', 'rb')
                # leaf_energy_last = pickle.load(fr)
                # fr.close()

                # not_end_leaf_idx = np.where(leaf_energy_last>Leaf_Ener_thrs)[0].tolist()#???
                # print("<===================== {} previous leaves has not reach the stopping energy thrs ======================>".format(len(not_end_leaf_idx)))
                # idx_list=[]
                # #///
                # idx_list = decide_pass_idx1(feature, leaf_energy_last, Leaf_Ener_thrs, Pass_Ener_thrs, hopidx=hopidx)
                # #///

                idx_list = idx_list   # fill in manually
                fwrite = open(weight_root+str(hopidx-1)+'/idx_to_pass.pkl', 'wb')
                pickle.dump(idx_list, fwrite, protocol=2)
                fwrite.close()

            else: # 1hop
                idx_list = np.arange(0,feature.shape[-1]).tolist()   # 1,3,5,6,7...
            print(idx_list)
            print("<===================== Pass {} previous leaves to this Hop ======================>".format(len(idx_list)))

            feature_ori = np.zeros((feature.shape[0], feature.shape[1], feature.shape[2], (2*window_size+1)**2, feature.shape[-1]))   # 10,256,384,25,28
            for c in idx_list:
                # feature_pool = PixelHop_batchSample(feature[:,:,:,[c]], indicator, dilate, pad) # (n,spacial,spect)
                feature_pool = PixelHop_Neighbour(feature[:,:,:,[c]], dilate, window_size, pad)

                feature_ori[:,:,:,:,c] = feature_pool[:,:,:,:,0]   # 10, 256, 384, 25

                feature_pool = feature_pool.reshape((-1,feature_pool.shape[-2],feature_pool.shape[-1]))
                n,sp,ch = feature_pool.shape
                print("       <PCA Info>        Pooled feature of PCA:",feature_pool.shape)
                dilate = np.array(dilate).astype('int64')
#                print("       <Info>        Kernel size: %s"%str(2*dilate[-1]+1))
#                print("       <Info>        Get covariance Sign: %s"%str(getcov))
                
                all_subTree['leaf'+str(c)] = treeSaab(weight_root + str(hopidx) +'/leaf' + str(c) +'.pkl',
                                                      kernel_sizes = np.array([2*window_size+1]),
                                                      num_kernels=num_kernels,
                                                      high_freq_percent= PCA_ener_percent,
                                                      getcov = getcov,
                                                      useDC= useDC)
                
                all_subTree['leaf'+str(c)].fit(feature_pool[:,:,0])

        else:
            # feature_pool = PixelHop_batchSample(feature, indicator, dilate, pad) # (n,spacial,spect)

            feature_pool = PixelHop_Neighbour(feature, dilate, window_size, pad)

            feature_ori = feature_pool

            feature_pool = feature_pool.reshape((-1,feature_pool.shape[-2],feature_pool.shape[-1]))   # ***, 25, 3
            n,sp,ch = feature_pool.shape
            print("       <PCA Info>        Pooled feature of PCA:",feature_pool.shape)   # *** ,25,3
            dilate = np.array(dilate).astype('int64')
#            print("       <Info>        Kernel size: %s"%str(2*dilate[-1]+1))
#            print("       <Info>        Get covariance Sign: %s"%str(getcov))
            all_subTree['leaf0'] = treeSaab(weight_root + str(hopidx) +'/leaf0.pkl',
                                            kernel_sizes = np.array([2*window_size+1]),
                                            high_freq_percent= PCA_ener_percent,
                                            getcov = getcov,
                                            useDC=useDC)
            # a = np.arange(600000)
            # b = np.random.permutation(a)
            all_subTree['leaf0'].fit(feature_pool.reshape(n,sp*ch))   # ***,75


        del feature_pool
        # all_subTree['total_leaf_num'] = int(total_leaf_num)
        fwrite = open(weight_root+str(hopidx)+'/all_subTree.pkl', 'wb')
        pickle.dump(all_subTree, fwrite, protocol=2)
        fwrite.close()

        # leaf_energy = np.concatenate(leaf_energy,axis=0).squeeze()
        #
        # print("Hop " + str(hopidx) + " leaf energy:", leaf_energy)

        # fwrite = open(weight_root+str(hopidx)+'/leaf_energy.pkl', 'wb')
        # pickle.dump(leaf_energy, fwrite, protocol=2)
        # fwrite.close()
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
                fread = open(weight_root + str(hopidx - 1) + '/idx_to_pass.pkl', 'rb')
                idx_list = pickle.load(fread)
                fread.close()
            else:  # 1hop
                idx_list = np.arange(0, feature.shape[-1]).tolist()
            print("<===================== Pass {} previous leaves to this Hop ======================>".format(
                len(idx_list)))
            feature_ori = np.zeros(
                (feature.shape[0], feature.shape[1], feature.shape[2], (2 * window_size + 1)**2, feature.shape[-1]))

            for c in idx_list:
                # feature_pool = PixelHop_batchSample(feature[:,:,:,[c]], indicator, dilate, pad) # (n,spacial,spect)
                feature_pool = PixelHop_Neighbour(feature[:, :, :, [c]], dilate, window_size, pad)

                feature_ori[:, :, :, :, c] = feature_pool[:, :, :, :, 0]
        else:
            # feature_pool = PixelHop_batchSample(feature, indicator, dilate, pad) # (n,spacial,spect)
            feature_pool = PixelHop_Neighbour(feature, dilate, window_size, pad)

            feature_ori = feature_pool

    print("=========== End: PixelHop_Unit -> using %10f seconds"%(time.time()-t0))

    return feature_ori


def PixelHop_fit(weight_name, feature_ori, split_spec=1,hopidx=1,Pass_Ener_thrs=0.2,Leaf_Ener_thrs=0.001):
    print("------------------- Start: Pixelhop_fit")
    print("       <Fit Info>        Using weight: %s"%str(weight_name))
    t0 = time.time()
    fread = open(weight_name+str(hopidx)+'/all_subTree.pkl', 'rb')
    all_subTree = pickle.load(fread)
    fread.close()
    n,x,y,sp,ch = feature_ori.shape
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
        if hopidx>1:
            fread = open(weight_name+str(hopidx-1)+'/idx_to_pass.pkl', 'rb')
            idx_list = pickle.load(fread)
            fread.close()    
        else:
            idx_list = np.arange(0,ch).tolist()
        for c in idx_list:
            # response[:,:,:,all_subTree['leaf'+str(c)]]
            _,temp_biased = all_subTree['leaf'+str(c)].transform(feature_ori[:,:,:,:,c],Bias=hopidx-1)# 1-Hop doesn't use bias
            # temp_biased shape: n, x, y, reduced_sp
            transformed_feature_biased.append(temp_biased)

            # first 3 pass, the rest stay
            temp_biased_pass = temp_biased[:,:,:,:3]
            temp_biased_stay = temp_biased[:,:,:,3:]

            pass_to_next_hop.append(temp_biased_pass)
            stay_at_current_hop.append(temp_biased_stay)


            # # show response
            # for sp in range(temp_biased.shape[-1]):
            #     plt.figure(0)
            #     plt.imshow(temp_biased[0,:,:,sp], cmap='coolwarm')
            #     plt.colorbar()
            #     plt.savefig('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/' + 'DEBUG/response/res_%d_%d.png' % (c, sp))
            #     plt.close(0)

        pass_to_next_hop = np.concatenate(pass_to_next_hop, axis=-1)
        stay_at_current_hop = np.concatenate(stay_at_current_hop, axis=-1)
        transformed_feature_biased = np.concatenate(transformed_feature_biased,axis=-1)
    else:
        transformed_feature, transformed_feature_biased = all_subTree['leaf0'].transform(feature_ori.reshape(n,x,y,sp*ch),Bias=hopidx-1)
        # # show response
        # for sp in range(transformed_feature_biased.shape[-1]):
        #     plt.figure(0)
        #     plt.imshow(transformed_feature_biased[0, :, :, sp], cmap='coolwarm')
        #     plt.colorbar()
        #     plt.savefig(
        #         '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/' + 'DEBUG/response/res_%d_%d.png' % (
        #         0, sp))
        #     plt.close(0)

    print("       <Fit Info>        Transformed feature shape for Hop: {}: %s".format(hopidx) %str(transformed_feature_biased.shape))
    print("------------------- End: Pixelhop_fit -> using %10f seconds"%(time.time()-t0))
    return pass_to_next_hop, stay_at_current_hop, transformed_feature_biased


def PixelHop_Aggregation(feat_map,win=5,agg_mode=None):
    # 1: max; 2:min; 3:mean
    fn,fh,fw,fc = feat_map.shape
    agg_len = len(agg_mode)
    agg_featmap = np.empty((fn,fh,fw,(agg_len+1)*fc))
    agg_featmap[:,:,:,:fc] = feat_map
    for x in range(fn):
        samples = window_process(feat_map[x].reshape(1,fh,fw,fc),win,1).squeeze().transpose(0,1,3,2)
        for k in range(len(agg_mode)):
            if agg_mode[k]==1:
                agg_featmap[x,:,:,fc*(k+1):fc*(k+2)] = np.max(samples,axis=-1)
            elif agg_mode[k]==2:
                agg_featmap[x,:,:,fc*(k+1):fc*(k+2)] = np.min(samples,axis=-1)
            elif agg_mode[k]==3:
                agg_featmap[x,:,:,fc*(k+1):fc*(k+2)] = np.mean(samples,axis=-1)
                
    return agg_featmap
