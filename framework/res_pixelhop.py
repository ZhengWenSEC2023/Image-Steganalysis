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

import numpy as np 
import pickle
import time
import os
from skimage.util.shape import view_as_windows
from framework.saab import Saab
from framework.residuePCA import ResPCA

    
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
    thrs=0.05
    indicator = np.empty(gt.shape)
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
    for i in range(int(feature.shape[0]/5)):
        temp = PixelHop_Neighbour(feature[i*5:(i+1)*5,:,:,:], dilate, pad)
        temp = temp.reshape((-1,temp.shape[-2],temp.shape[-1]))
        print('batchSample temp shape:', temp.shape)
        if i == 0:
            feature_pool = np.empty((np.sum(indicator),temp.shape[-2], temp.shape[-1]))
            START = 0
        END = np.sum(indicator[:(i+1)*5])
        feature_pool[START:END] = np.copy(temp[indicator[i*5:(i+1)*5].reshape(-1)==1])
        del temp
        START = np.copy(END)
#    feature_pool = feature_pool.reshape(feature_pool.shape[0]*feature_pool.shape[1], -1)
    print(">>>>>>>>>>>>>>>>>>> finish batch sampling")
    return feature_pool
    
    
def PixelHop_Neighbour(feature, dilate, pad):
    print("------------------- Start: PixelHop_Neighbour")
    print("       <Info>        Input feature shape: %s"%str(feature.shape))
    print("       <Info>        dilate: %s"%str(dilate))
    print("       <Info>        padding: %s"%str(pad))
    t0 = time.time()
    S = feature.shape
    idx = [-1, 0, 1]
    if pad == 'reflect':
        feature = np.pad(feature, ((0,0),(dilate[-1], dilate[-1]),(dilate[-1], dilate[-1]),(0,0)), 'reflect')
    elif pad == 'zeros':
        feature = np.pad(feature, ((0,0),(dilate[-1], dilate[-1]),(dilate[-1], dilate[-1]),(0,0)), 'constant', constant_values=0)
    if pad == "none":
        res = np.zeros((S[1]-2*dilate[-1], S[2]-2*dilate[-1], S[0], 9*S[3]))
    else:
        dilate = np.array(dilate).astype('int64')
        if dilate[0]==0:
#            if dilate[1]==1:# if dilate[0]+dilate[1] = 0, then extract patch
            res = np.zeros((S[1], S[2], S[0], (2*dilate[-1]+1)**2, S[3]))
#            else:
#                res = np.zeros((S[1], S[2], S[0], (8*(dilate.shape[0]-1)+1), S[3]))          
        else:
            res = np.zeros((S[1], S[2], S[0], (8*dilate.shape[0]+1), S[3]))
    feature = np.moveaxis(feature, 0, 2)
    for i in range(dilate[-1], feature.shape[0]-dilate[-1]):
        for j in range(dilate[-1], feature.shape[1]-dilate[-1]):
            if dilate[0]==0:
                d = dilate[-1]
                tmp = np.copy(feature[(i+idx[0]*d):(i+idx[-1]*d+1),(j+idx[0]*d):(j+idx[-1]*d+1)])
#                tmp = tmp[::dilate[1],::dilate[1]]
#                tmp = tmp.reshape(-1, tmp.shape[-2], tmp.shape[-1])                
                tmp = tmp.reshape((2*dilate[-1]+1)**2, tmp.shape[-2], tmp.shape[-1])
            else:
                tmp = []
                for d in dilate:
                    for ii in idx:
                        for jj in idx:
                            if ii == 0 and jj == 0:
                                continue
                            iii = i+ii*d
                            jjj = j+jj*d
                            tmp.append(feature[iii, jjj])
                tmp.append(feature[i,j])
                tmp = np.array(tmp)
            tmp = np.moveaxis(tmp,0,1)
            res[i-dilate[-1], j-dilate[-1]] = np.copy(tmp)
    res = np.moveaxis(res, 2, 0)
    # res = np.moveaxis(res, -2, -1)
    print("       <Info>        Output feature shape: %s"%str(res.shape))
    print("------------------- End: PixelHop_Neighbour -> using %10f seconds"%(time.time()-t0))
    return res 


def PixelHop_Unit(feature, dilate=np.array([2]), train = True, pad='reflect', weight_root='./tmp', Pass_Ener_thrs=0.2, energy_percent = 0.98, useDC=False, stride = None, getcov=0, split_spec=1, hopidx=1):
    print("=========== Create weight root ...")    
    if not os.path.isdir(weight_root+str(hopidx)+'/'):os.makedirs(weight_root+str(hopidx)+'/')    
    print("=========== Start: PixelHop_Unit")
    t0 = time.time()
    if train == True:
        print("----------- Training -----------")
        print(">>>>>====== PixelHop neighbor construction")
        # indicator = sample_indicator(gt)
        total_leaf_num = 0.0
        all_subTree = {}
        leaf_energy = []
        feature_ori = np.zeros((feature.shape[0], feature.shape[1], feature.shape[2], 9, feature.shape[-1]))
        if split_spec == 1 :
            if hopidx>1:
                fr = open(weight_root+str(hopidx-1)+'/leaf_energy.pkl', 'rb')
                leaf_energy_last = pickle.load(fr)
                fr.close()
                idx_list = np.where(leaf_energy_last>Pass_Ener_thrs)[0].tolist()#???
            else:
                idx_list = np.arange(0,feature.shape[-1]).tolist()
            print("<===================== Pass {} previous leaves to this Hop ======================>".format(len(idx_list)))
            for c in idx_list:
                # feature_pool = PixelHop_batchSample(feature[:,:,:,[c]], indicator, dilate, pad) # (n,spacial,spect)
                feature_pool = PixelHop_Neighbour(feature[:,:,:,[c]], dilate, pad)
                print("feature pool shape:", feature_pool.shape)
                feature_ori[:,:,:,:,c] = feature_pool[:,:,:,:,0]

                feature_pool = feature_pool.reshape((-1,feature_pool.shape[-2],feature_pool.shape[-1]))


                n,sp,ch = feature_pool.shape

                print(">>>>>====== Channel wise PCA")

                print("       <Info>        Pooled feature of PCA:",feature_pool.shape)
                dilate = np.array(dilate).astype('int64')
                print("       <Info>        Kernel size: %s"%str(2*dilate[-1]+1))
                print("       <Info>        Get covariance Sign: %s"%str(getcov))

                all_subTree['leaf'+str(c)] = ResPCA(weight_root+str(hopidx)+'/leaf'+str(c)+'.pkl',
                                                    kernel_sizes = np.array([2*dilate[-1]+1]),
                                                    target_ener_percent = energy_percent,
                                                    getcov = getcov)

                all_subTree['leaf'+str(c)].fit(feature_pool[:,:,0])
                # allTree['leaf'+str(c)] = rpca
                total_leaf_num += all_subTree['leaf'+str(c)].leaf_num
                leaf_energy.append(all_subTree['leaf'+str(c)].energy)

        else:
            # feature_pool = PixelHop_batchSample(feature, indicator, dilate, pad) # (n,spacial,spect)
            feature_pool = PixelHop_Neighbour(feature, dilate, pad)
            feature_ori = feature_pool

            feature_pool = feature_pool.reshape((-1,feature_pool.shape[-2],feature_pool.shape[-1]))
            n,sp,ch = feature_pool.shape

            print(">>>>>====== Channel wise PCA")

            print("       <Info>        Pooled feature of PCA:",feature_pool.shape)
            dilate = np.array(dilate).astype('int64')
            print("       <Info>        Kernel size: %s"%str(2*dilate[-1]+1))
            print("       <Info>        Get covariance Sign: %s"%str(getcov))
            all_subTree['leaf0'] = ResPCA(weight_root+str(hopidx)+'/leaf0.pkl',
                                                kernel_sizes = np.array([2*dilate[-1]+1]),
                                                target_ener_percent = energy_percent,
                                                getcov = getcov)
            all_subTree['leaf0'].fit(feature_pool.reshape(n,sp*ch))
            # allTree['leaf'+str(c)] = rpca
            total_leaf_num += all_subTree['leaf0'].leaf_num
            leaf_energy = all_subTree['leaf0'].energy

        # del feature_pool


        all_subTree['total_leaf_num'] = int(total_leaf_num)
        fw = open(weight_root+str(hopidx)+'/all_subTree.pkl', 'wb')
        pickle.dump(all_subTree, fw, protocol=2)
        fw.close()
        leaf_energy = np.concatenate(leaf_energy,axis=0).squeeze()
        fw = open(weight_root+str(hopidx)+'/leaf_energy.pkl', 'wb')
        pickle.dump(leaf_energy, fw, protocol=2)
        fw.close()
        print("=========== Final primal leaf nodes: {}".format(total_leaf_num))


    else:
        print("----------- Testing -----------")
        print(">>>>>====== PixelHop neighbor construction")
        # indicator = sample_indicator(gt)
        feature_ori = np.zeros(
            (feature.shape[0], feature.shape[1], feature.shape[2], 9, feature.shape[-1]))
        if split_spec == 1:
            if hopidx > 1:
                fr = open(weight_root + str(hopidx - 1) + '/leaf_energy.pkl', 'rb')
                leaf_energy_last = pickle.load(fr)
                fr.close()
                idx_list = np.where(leaf_energy_last > Pass_Ener_thrs)[0].tolist()  # ???
            else:
                idx_list = np.arange(0, feature.shape[-1]).tolist()
            print("<===================== Pass {} previous leaves to this Hop ======================>".format(
                len(idx_list)))
            for c in idx_list:
                # feature_pool = PixelHop_batchSample(feature[:,:,:,[c]], indicator, dilate, pad) # (n,spacial,spect)
                feature_pool = PixelHop_Neighbour(feature[:, :, :, [c]], dilate, pad)
                print("feature pool shape:", feature_pool.shape)
                feature_ori[:, :, :, :, c] = feature_pool[:, :, :, :, 0]

                # feature_pool = feature_pool.reshape((-1, feature_pool.shape[-2], feature_pool.shape[-1]))

                # n, sp, ch = feature_pool.shape
        else:
            # feature_pool = PixelHop_batchSample(feature, indicator, dilate, pad) # (n,spacial,spect)
            feature_pool = PixelHop_Neighbour(feature, dilate, pad)
            feature_ori = feature_pool

            # feature_pool = feature_pool.reshape((-1, feature_pool.shape[-2], feature_pool.shape[-1]))
            # n, sp, ch = feature_pool.shape

    print("=========== End: PixelHop_Unit -> using %10f seconds"%(time.time()-t0))

    return feature_ori


def PixelHop_fit(weight_name, feature_ori, split_spec=1,hopidx=1,Pass_Ener_thrs=0.2):
    print("------------------- Start: Pixelhop_fit")
    print("       <Info>        Using weight: %s"%str(weight_name))
    t0 = time.time()
    fr = open(weight_name+str(hopidx)+'/all_subTree.pkl', 'rb')
    all_subTree = pickle.load(fr)
    fr.close()
    n,x,y,sp,ch = feature_ori.shape
    # response = np.zeros(feature_ori.shape[0], feature_ori.shape[1], feature_ori.shape[2], all_subTree['total_leaf_num'])
    transformed_feature_biased = []
    if split_spec == 1:
        if hopidx>1:
            fr = open(weight_name+str(hopidx-1)+'/leaf_energy.pkl', 'rb')
            leaf_energy = pickle.load(fr)
            fr.close()
            idx_list = np.where(leaf_energy>Pass_Ener_thrs)[0].tolist()
        else:
            idx_list = np.arange(0,ch).tolist()
        for c in idx_list:
            # response[:,:,:,all_subTree['leaf'+str(c)]]
            temp_biased = all_subTree['leaf'+str(c)].transform(feature_ori[:,:,:,:,c])
            print("fit temp biased shape:", temp_biased.shape)
            transformed_feature_biased.append(temp_biased)

        transformed_feature_biased = np.concatenate(transformed_feature_biased,axis=-1)
    else:
        transformed_feature_biased = all_subTree['leaf0'].transform(feature_ori.reshape(n,x,y,sp*ch))
    
    print("       <Info>        Transformed feature shape: %s"%str(transformed_feature_biased.shape))
    print("------------------- End: Pixelhop_fit -> using %10f seconds"%(time.time()-t0))
    return transformed_feature_biased


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
