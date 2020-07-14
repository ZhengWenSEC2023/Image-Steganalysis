import os
import sys
import numpy as np
import time
import pickle
import scipy
import sklearn
import math
import random
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

# from regression import myRegression
import warnings

warnings.filterwarnings('ignore')


def threshold(img, lowThreshold = -0.3, highThreshold = 0.3):

    output_prob_map = []
    weak_loc = [[] for i in range(img.shape[0])]

    for k in range(img.shape[0]):

        M, N = img[k].shape
        res = np.zeros((M, N))

        strong_authentic = -1
        strong_spliced = 1

        strong_i, strong_j = np.where(img[k] >= highThreshold)
        zeros_i, zeros_j = np.where(img[k] < lowThreshold)

        weak_i, weak_j = np.where((img[k] <= highThreshold) & (img[k] >= lowThreshold))

        res[strong_i, strong_j] = strong_spliced
        res[zeros_i, zeros_j] = strong_authentic
        res[weak_i, weak_j] = img[k,weak_i, weak_j]

        output_prob_map.append(res)
        weak_loc[k].append([weak_i, weak_j])




    output_prob_map = np.array(output_prob_map)

    return output_prob_map, weak_loc


def Regression_Method(X, Y):

    reg = RandomForestRegressor(max_depth=None, n_estimators=100, random_state=0, min_samples_split= 400, min_samples_leaf=200)
    # reg = LinearRegression()
    # reg = LogisticRegression(solver='saga', multi_class='ovr', n_jobs=8)   # this is a classifier

    # reg = RandomForestClassifier(verbose=0, class_weight='balanced',
    #                                   n_estimators=30, min_samples_split=2,
    #                                   max_features='auto', bootstrap=True,
    #                                   max_depth=5, min_samples_leaf=2)

    reg.fit(X, Y)

    return reg


# def Majority_Vote(Y, mvth):
#     new_label = -1
#     label = np.unique(Y)
#     for i in range(label.shape[0]):
#         if Y[Y == label[i]].shape[0] > mvth * (float)(Y.shape[0]):
#             new_label = label[i]
#             break
#     return new_label


# select next leaf node to be splited
# alpha: weight importance
def Select_Next_Split(data, Hidx, alpha):
    t = 0
    idx = 0
    for i in range(0, len(Hidx) - 1): # 0,1
        # tt = data[Hidx[i]]['H']*np.exp(-1*alpha/(float)(data[Hidx[i]]['Data'].shape[0]))
        tt = data[Hidx[i]]['H_val'] * np.log((float)(data[Hidx[i]]['Data'].shape[0])) / np.log(alpha + 1)
        # child node 0 ce * log(sample num in child node 0) /log(2)
        if t < tt:
            t = tt
            idx = i
    return idx


################################# Global Cross Entropy #################################
def Compute_GlobalH(X, total, Hidx):
    gH_val = 0.0
    gH_train = 0.0
    H_val = []
    H_train = []
    w = []
    for i in range(len(Hidx) - 1):   # 0, 1  Hidx = [1, 2, 3]    X = data = [{root node}, {child node 0}, {child node 1}]
        w.append(X[Hidx[i]]['Data'].shape[0] / float(total))   # w: [child node 0/ total, child node 1/ total]
        H_val.append(X[Hidx[i]]['H_val'])   # cross entropy of child node 1 & 2
        H_train.append(X[Hidx[i]]['H_train'])

        gH_val += (X[Hidx[i]]['Data'].shape[0] / float(total)) * X[Hidx[i]]['H_val']
        gH_train += (X[Hidx[i]]['Data'].shape[0] / float(total)) * X[Hidx[i]]['H_train']

    print("       <Debug Info>        MSE: %s" % str(H_val))
    print("       <Debug Info>        Weight: %s" % str(w))
    return gH_val, gH_train



################################# Cross Entropy #################################
# used when computing entropy in <Multi_Trial>
def Compute_Weight(Y, num_class):
    weight = np.zeros(num_class)     # 2,

    if (Y[Y <= 0].shape[0]) == 0:
        weight[0] = 0
    else:
        weight[0] = (float)(Y[Y <= 0].shape[0])

    if (Y[Y > 0].shape[0]) == 0:
        weight[1] = 0
    else:
        weight[1] = (float)(Y[Y > 0].shape[0])

    weight /= np.sum(weight)
    return weight


# latest cross entropy method
def Comupte_mse(X, Y_target):
    samp_num = Y_target.size
    # if X.shape[0] < num_bin:     # sample num in cluster is too small!
    #     return -1

    # split 80% and 20% training and validation set
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y_target, test_size = 0.2, random_state = 42)

    # train classifier on X_train, Y_train
    reg = RandomForestRegressor(n_estimators=50, min_samples_split=2, max_features='auto',
                           bootstrap=True, max_depth=5, min_samples_leaf=2)
    reg.fit(X_train, Y_train)

    Y_val_predicted = reg.predict(X_val)
    Y_train_predicted = reg.predict(X_train)


    train_loss = metrics.mean_squared_error(Y_train, Y_train_predicted)
    val_loss = metrics.mean_squared_error(Y_val, Y_val_predicted)
    return val_loss, train_loss


################################# Init For Root Node #################################
# init kmeans centroid with center of samples from each label, then do kmeans
def Init_By_Class(X, Y_target, num_class, sep_num, trial):
    init_centroid = []

    Y = Y_target.reshape(-1)
    init_centroid.append(np.mean(X[Y <= 0], axis=0).reshape(-1))
    init_centroid.append(np.mean(X[Y > 0], axis=0).reshape(-1))

    tmpH = 100000
    init_centroid = np.array(init_centroid)

    for i in range(trial):
        t = np.arange(0, num_class).tolist()    # 0,1
        tmp_idx = np.array(random.sample(t, len(t)))
        km = KMeans(n_clusters=sep_num, init=init_centroid[tmp_idx[0:sep_num]]).fit(X)

        ce = np.zeros((sep_num))
        w = np.zeros((sep_num))
        t_totalH = 0.0
        for j in range(sep_num):  # 0, 1
            ce[j] = Comupte_mse(X[km.labels_ == j], Y[km.labels_ == j])[0]
            w[j] = X[km.labels_ == j].shape[0] / Y.shape[0]
            t_totalH += w[j]*ce[j]

        if tmpH > t_totalH:
            kmeans = km
            tmpH = t_totalH

    data = []
    mse_X = Comupte_mse(X, Y)
    data.append({'Data': [],
                 'Target': [],
                 'Centroid': np.mean(X, axis=1),
                 'H_val': mse_X[0],
                 'H_train': mse_X[1],
                 'ID': str(-1)})
    H = []
    Hidx = [1]
    for i in range(sep_num):    # 0, 1

        mses = Comupte_mse(X[kmeans.labels_ == i], Y[kmeans.labels_ == i])
        data.append({'Data': X[kmeans.labels_ == i],
                     'Target': Y[kmeans.labels_ == i],
                     'Centroid': kmeans.cluster_centers_[i],
                     'H_val': mses[0],
                     'H_train': mses[1],
                     'ID': str(i)})
        H.append(data[i]['H_val'])        # root node ce, child node 1 ce, child node 2 ce
        Hidx.append(Hidx[-1] + 1)     # 1, 2, 3

    # X0 = data[1]['Data']
    # X1 = data[2]['Data']
    # Y0 = data[1]['Label']
    # Y1 = data[2]['Label']

    # data[1]['Data'] = []
    # data[2]['Data'] = []
    # data[1]['Label'] = []
    # data[2]['Label'] = []


    # km0 = KMeans(n_clusters=5, init='random').fit(X0)
    # km1 = KMeans(n_clusters=5, init='random').fit(X1)
    # for i in range(5):
    #     data.append({'Data': X0[km0.labels_ == i],
    #                 'Label': Y0[km0.labels_ == i],
    #                 'Centroid': km0.cluster_centers_[i],
    #                 'H_val': Comupte_Cross_Entropy(X0[km0.labels_ == i], Y0[km0.labels_ == i], num_class)[0],
    #                 'H_train': Comupte_Cross_Entropy(X0[km0.labels_ == i], Y0[km0.labels_ == i], num_class)[1],
    #                 'ID': data[1]['ID'] + str(i)})
    #     H.append(data[i+3]['H_val'])
    #     Hidx.append(Hidx[-1] + 1)
    #
    # print(len(data))
    #
    #
    # for i in range(5):
    #     data.append({'Data': X1[km1.labels_ == i],
    #                 'Label': Y1[km1.labels_ == i],
    #                 'Centroid': km1.cluster_centers_[i],
    #                 'H_val': Comupte_Cross_Entropy(X1[km1.labels_ == i], Y1[km1.labels_ == i], num_class)[0],
    #                 'H_train': Comupte_Cross_Entropy(X1[km1.labels_ == i], Y1[km1.labels_ == i], num_class)[1],
    #                 'ID': data[2]['ID'] + str(i)})
    #     H.append(data[i + 8]['H_val'])
    #     Hidx.append(Hidx[-1] + 1)


    return data, H, Hidx


# # init whole data as one cluster
# def Init_As_Whole(X, Y, num_class):
#     data = [{'Data': X,
#              'Label': Y,
#              'Centroid': np.mean(X, axis=0),
#              'H_val': Comupte_Cross_Entropy(X, Y, num_class)[0],
#              'H_train': Comupte_Cross_Entropy(X, Y, num_class)[1],
#              'ID': '0'}]
#     H = [data[0]['H_val']]
#     Hidx = [0, 1]
#     return data, H, Hidx


################################# KMeans Init Methods #################################
# LBG initialization
def Init_LBG(X, sep_num):
    c1 = np.mean(X, axis=0).reshape(1, -1)
    st = np.std(X, axis=0)
    dic = {}
    new_centroid = c1
    for i in range(sep_num - 1):
        n = np.random.randint(2, size=X.shape[1])
        n[n == 0] = -1
        if str(n) in dic:
            continue
        dic[str(n)] = 1
        c2 = c1 + n * st
        new_centroid = np.concatenate((new_centroid, c2.reshape(1, -1)), axis=0)
    return new_centroid


################################# Loss_Guided_KMeans train #################################
# try multiply times when spliting the leaf node
def Multi_Trial(X, sep_num, batch_size, trial, num_class, err):
    init = ['k-means++', 'random', 'k-means++', 'random', 'k-means++', 'random', Init_LBG(X['Data'], sep_num)]
    H = X['H_val'] - err
    center = []
    t_entropy = np.zeros((trial + 1))   # 4  total child node cross entropy for every trial
    t_entropy[-1] = H + err
    for i in range(trial): # 0,1,2
        if batch_size == None:
            kmeans = KMeans(n_clusters=sep_num, init=init[i % len(init)]).fit(X['Data'])
        else:
            kmeans = MiniBatchKMeans(n_clusters=sep_num, batch_size=batch_size).fit(X['Data'])
        weight = Compute_Weight(kmeans.labels_, num_class)

        for k in range(sep_num):
            t_entropy[i] += weight[k] * Comupte_mse(X['Data'][kmeans.labels_ == k],
                                                              X['Target'][kmeans.labels_ == k])[0]
        if t_entropy[i] < H:
            H = t_entropy[i]
            center = kmeans.cluster_centers_.copy()
            label = kmeans.labels_.copy()
            print("           <Info>        Multi_Trial %s: Found a separation better than original! CE: %s" % (
            str(i), str(H)))
    print("           <Debug Info>        Global entropy of each trial %s: " % (str(t_entropy)))
    if len(center) == 0:
        return []
    subX = []
    for i in range(sep_num):
        idx = (label == i)
        mse_sub = Comupte_mse(X['Data'][idx], X['Target'][idx])
        subX.append({'Data': X['Data'][idx], 'Target': X['Target'][idx], 'Centroid': center[i],
                     'H_val': mse_sub[0],
                     'H_train': mse_sub[1],
                     'ID': X['ID'] + str(i)})
    return subX


def Leaf_Node_Regression(data, Hidx, num_class):
    for i in range(len(Hidx) - 1):
        data[Hidx[i]]['Regressor'] = Regression_Method(data[Hidx[i]]['Data'], data[Hidx[i]]['Target'])
        data[Hidx[i]]['Data'] = []  # no need to store raw data any more
        data[Hidx[i]]['Target'] = []
    return data


def Loss_Guided_KMeans_train(X, Y, sep_num, trial, batch_size, minS, maxN, err, mvth, maxdepth, alpha):
    print("------------------- Start: Loss_Guided_KMeans_train")
    t0 = time.time()
    print("       <Info>        Trial: %s" % str(trial))
    print("       <Info>        Batch size: %s" % str(batch_size))
    print("       <Info>        Minimum number of samples in each cluster: %s" % str(minS))
    print("       <Info>        Max number of leaf nodes: %s" % str(maxN))
    print("       <Info>        Stop splitting: %s" % str(err))
    print("       <Info>        Max depth: %s" % str(maxdepth))
    print("       <Info>        Alpha: %s" % str(alpha))
    # H: <list> entropy of nodes can be split
    # Hidx: <list> location of corresponding H in data
    # num_class = np.unique(Y).shape[0]
    num_class = 2
    data, H, Hidx = Init_By_Class(X, Y, num_class, sep_num, trial)    # data: list 里面多个dict, 每一个node都是一个dict
    rootSampNum = Y.shape[0]    # total number of samples in root
    global_H_val = []
    global_H_train = []

    X, Y = [], []
    N, myiter = 1, 1
    print("\n       <Info>        Start iteration")
    print("       <Info>        Iter %s" % (str(0)))
    global_mse = Compute_GlobalH(data, rootSampNum, Hidx)
    global_H_val.append(global_mse[0])
    global_H_train.append(global_mse[1])

    while N < maxN:
        print("       <Info>        Iter %s" % (str(myiter)))
        idx = Select_Next_Split(data, Hidx, alpha)
        print("           <Debug>       Hidx:", Hidx)
        print("           <Debug>       which idx to split:", idx)
        print("           <Debug>       Global MSE val:", global_H_val)
        print("           <Debug>       Global MSE train:", global_H_train)

        # finish splitting, when no node need further split
        if H[idx] <= 0:
            print("       <Info>        Finish splitting!")
            break

        # if this cluster has too few sample, do not split this node
        if data[Hidx[idx]]['Data'].shape[0] < minS:
            print("       <Warning>        Iter %s: Too small! continue for the next largest" % str(myiter))
            H[idx] = -H[idx]
            continue

        # maxdepth
        if len(data[Hidx[idx]]['ID']) >= maxdepth:
            print("       <Warning>        Depth >= maxdepth %s: Too small! continue for the next largest" % str(
                maxdepth))
            H[idx] = -H[idx]
            continue

        # # majority vote
        # tmp = Majority_Vote(data[Hidx[idx]]['Label'], mvth)
        # if tmp != -1:
        #     print("       <Warning>        Majority vote on this node, no further split needed")
        #     H[idx] = -H[idx]
        #     data[Hidx[idx]]['Label'] = tmp * np.ones((data[Hidx[idx]]['Label'].shape[0]))
        #     continue

        # try to split this node multi times

        subX = Multi_Trial(data[Hidx[idx]], sep_num=sep_num, batch_size=batch_size, trial=trial, num_class=num_class,
                           err=err)

        # find a better splitting?
        if len(subX) != 0:
            # save memory, do not save X, Y multi times
            data[Hidx[idx]]['Data'] = []      # 清空 parent node data & label
            data[Hidx[idx]]['Target'] = []
            data += subX    # 加入两个child node {} 信息
            H.pop(idx)
            Hidx.pop(idx)
            N -= 1
            for d in subX:
                H.append(d['H_val'])
                Hidx.append(Hidx[-1] + 1)
                N += 1
            myiter += 1
            global_mse = Compute_GlobalH(data, rootSampNum, Hidx)
            global_H_val.append(global_mse[0])
            global_H_train.append(global_mse[1])

        else:
            print("       <Warning>        Iter %s: Don't split! continue for the next largest" % str(myiter))
            H[idx] = -H[idx]
    # data = Leaf_Node_Regression(data, Hidx, num_class)
    print("------------------- End: Loss_Guided_KMeans_train -> using %10f seconds" % (time.time() - t0))
    return data, global_H_val


################################# Loss_Guided KMeans Test #################################
# list to dictionary
def List2Dict(data):
    res = {}
    for i in range(len(data)):
        res[data[i]['ID']] = data[i]
    return res


def Loss_Guided_KMeans_Iter_test(X, key_parent, data, sep_num):
    centroid = []
    key_child = []
    for i in range(sep_num):
        if key_parent + str(i) in data:
            centroid.append(data[key_parent + str(i)]['Centroid'].reshape(-1))
            key_child.append(key_parent + str(i))
    centroid = np.array(centroid)
    dist = euclidean_distances(X.reshape(1, -1), centroid).squeeze()
    key = key_child[np.argmin(dist)]
    if 'Regressor' in data[key]:
        return key
    return Loss_Guided_KMeans_Iter_test(X, key, data, sep_num)


def Loss_Guided_KMeans_test(X, data, sep_num):
    for i in range(X.shape[0]):
        if i == 0:
            pred = data[Loss_Guided_KMeans_Iter_test(X[i], '', data, sep_num)]['Regressor'].predict_proba(X[i].reshape(1, -1))
        else:
            pred = np.concatenate((pred, data[Loss_Guided_KMeans_Iter_test(X[i], '', data, sep_num)]['Regressor'].predict_proba(
                X[i].reshape(1, -1))), axis=0)
    return pred


################################# MAIN Function #################################
def Loss_Guided_KMeans(X, Y=None, path='Tree/Kmeantree.pkl', train=True, sep_num=2, trial=3, batch_size=1000, minS=100, maxN=50,
               err=0.005, mvth=0.99, maxdepth=50, alpha=1):
    print("=========== Start: Loss_Guided_KMeans")
    print("       <Info>        Input shape: %s" % str(X.shape))
    print("       <Info>        train: %s" % str(train))
    t0 = time.time()
    if train == True:
        data, globalH = Loss_Guided_KMeans_train(X, Y, sep_num=sep_num, trial=trial, batch_size=batch_size, minS=minS,
                                         maxN=maxN, err=err, mvth=mvth, maxdepth=maxdepth, alpha=alpha)
        data = List2Dict(data)
        f = open(save_dir_matFiles + path, 'wb')
        pickle.dump(data, f)
        f.close()
    else:
        f = open(save_dir_matFiles + path, 'rb')
        data = pickle.load(f)
        f.close()
    # X = Loss_Guided_KMeans_test(X, data, sep_num)
    print("=========== End: Loss_Guided_KMeans_train -> using %10f seconds" % (time.time() - t0))
    return X, data



if __name__=='__main__':

    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/'
    print("----Training----")
    # features should be the same
    positive1 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/pos_1_100.npy')
    negative1 = pickle.load(open(save_dir_matFiles + 'nonoverlap_s1a1/neg_1_100.pkl', 'rb'))
    positive2 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/pos_2_100.npy')
    negative2 = pickle.load(open(save_dir_matFiles + 'nonoverlap_s1a1/neg_2_100.pkl', 'rb'))
    print("train shape:", positive2.shape, negative2.shape)

    X_1 = np.concatenate((positive1, negative1), axis=0)
    X_2 = np.concatenate((positive2, negative2), axis=0)

    # target is different with differnt weights
    pos_target_s1a1 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/pos_target.npy')
    neg_target_s1a1 = pickle.load(open(save_dir_matFiles + 'nonoverlap_s1a1/neg_target.pkl', 'rb'))
    print(np.max(pos_target_s1a1), np.min(pos_target_s1a1), np.max(neg_target_s1a1), np.min(neg_target_s1a1))

    pos_target_s1a2 = np.load(save_dir_matFiles + 'nonoverlap_s1a2/pos_target.npy')
    neg_target_s1a2 = pickle.load(open(save_dir_matFiles + 'nonoverlap_s1a2/neg_target.pkl', 'rb'))
    print(np.max(pos_target_s1a2), np.min(pos_target_s1a2), np.max(neg_target_s1a2), np.min(neg_target_s1a2))

    pos_target_s2a1 = np.load(save_dir_matFiles + 'nonoverlap_s2a1/pos_target.npy')
    neg_target_s2a1 = pickle.load(open(save_dir_matFiles + 'nonoverlap_s2a1/neg_target.pkl', 'rb'))
    print(np.max(pos_target_s2a1), np.min(pos_target_s2a1), np.max(neg_target_s2a1), np.min(neg_target_s2a1))


    # read test features, same for different weights
    test_positive1 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/columbia/test_pos_1_40.npy')
    test_negative1 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/columbia/test_neg_1_40.npy')
    test_positive2 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/columbia/test_pos_2_40.npy')
    test_negative2 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/columbia/test_neg_2_40.npy')
    print("test shape:", test_positive2.shape, test_negative2.shape)



    Y_target_s1a1 = np.concatenate((pos_target_s1a1, neg_target_s1a1), axis=0)
    Y_target_s1a2 = np.concatenate((pos_target_s1a2, neg_target_s1a2), axis=0)
    Y_target_s2a1 = np.concatenate((pos_target_s2a1, neg_target_s2a1), axis=0)




    #          ###########       TRAINING       ##############

    # split 80% and 20% training and validation set
    # X_train, X_val, Y_train, Y_val = train_test_split(X_1, Y_target, test_size=0.1, random_state=42)

    # X_train, X_val, Y_train, Y_val = train_test_split(X_2, Y_target, test_size=0.1, random_state=42)
    shuffle_idx = np.random.permutation(np.arange(X_2.shape[0]))

    flag = int(X_2.shape[0]*0.8)
    X_train = X_2[shuffle_idx[:flag], :]
    X_val = X_2[shuffle_idx[flag:], :]
    Y_train_s1a1 = Y_target_s1a1[shuffle_idx[:flag]]
    Y_val_s1a1 = Y_target_s1a1[shuffle_idx[flag:]]
    Y_train_s1a2 = Y_target_s1a2[shuffle_idx[:flag]]
    Y_val_s1a2 = Y_target_s1a2[shuffle_idx[flag:]]
    Y_train_s2a1 = Y_target_s2a1[shuffle_idx[:flag]]
    Y_val_s2a1 = Y_target_s2a1[shuffle_idx[flag:]]

    Y_train_label = np.ones((Y_train_s1a1.shape))
    for k in range(Y_train_s1a1.shape[0]):
        if Y_train_s1a1[k] < 0:
            Y_train_label[k] = 0
    Y_val_label = np.ones((Y_val_s1a1.shape))
    for k in range(Y_val_s1a1.shape[0]):
        if Y_val_s1a1[k] < 0:
            Y_val_label[k] = 0




    # train classifier on X_train, Y_train
    # reg = RandomForestRegressor(n_estimators=250, n_jobs=8, criterion='mse', min_samples_split=200, min_samples_leaf=100)
    # reg.fit(X_train, Y_train)
    #
    # with open(save_dir_matFiles + 'nonoverlap_5by5/rf_hop2.pkl', 'wb') as fid:
    #     pickle.dump(reg, fid)

    # with open(save_dir_matFiles + 'rf_hop1.pkl', 'rb') as fid:
    #     reg1 = pickle.load(fid)

    with open(save_dir_matFiles + 'nonoverlap_s1a1/rf_hop2.pkl', 'rb') as fid:
        reg_s1a1 = pickle.load(fid)
    with open(save_dir_matFiles + 'nonoverlap_s1a2/rf_hop2.pkl', 'rb') as fid:
        reg_s1a2 = pickle.load(fid)
    with open(save_dir_matFiles + 'nonoverlap_s2a1/rf_hop2.pkl', 'rb') as fid:
        reg_s2a1 = pickle.load(fid)




    Y_train_pre1 = reg_s1a1.predict(X_train).reshape(Y_train_s1a1.shape[0], 1)
    Y_train_pre2 = reg_s1a2.predict(X_train).reshape(Y_train_s1a2.shape[0], 1)
    Y_train_pre3 = reg_s2a1.predict(X_train).reshape(Y_train_s2a1.shape[0], 1)

    Y_val_pre1 = reg_s1a1.predict(X_val).reshape(Y_val_s1a1.shape[0], 1)
    Y_val_pre2 = reg_s1a2.predict(X_val).reshape(Y_val_s1a2.shape[0], 1)
    Y_val_pre3 = reg_s2a1.predict(X_val).reshape(Y_val_s2a1.shape[0], 1)

    print(np.max(Y_train_pre1), np.min(Y_train_pre1))
    print(np.max(Y_train_pre2), np.min(Y_train_pre2))
    print(np.max(Y_train_pre3), np.min(Y_train_pre3))


    # normalize 3 predicted target to [-1, 1]
    Y_train_pre1_norm = 2 * (Y_train_pre1 - np.min(Y_train_pre1)) / (np.max(Y_train_pre1) - np.min(Y_train_pre1)) + 1
    Y_train_pre2_norm = 2 * (Y_train_pre2 - np.min(Y_train_pre2)) / (np.max(Y_train_pre2) - np.min(Y_train_pre2)) + 1
    Y_train_pre3_norm = 2 * (Y_train_pre3 - np.min(Y_train_pre3)) / (np.max(Y_train_pre3) - np.min(Y_train_pre3)) + 1

    Y_val_pre1_norm = 2 * (Y_val_pre1 - np.min(Y_val_pre1)) / (np.max(Y_val_pre1) - np.min(Y_val_pre1)) + 1
    Y_val_pre2_norm = 2 * (Y_val_pre2 - np.min(Y_val_pre2)) / (np.max(Y_val_pre2) - np.min(Y_val_pre2)) + 1
    Y_val_pre3_norm = 2 * (Y_val_pre3 - np.min(Y_val_pre3)) / (np.max(Y_val_pre3) - np.min(Y_val_pre3)) + 1

    #%%
    # fuse predicted results from three weights target
    Y_train_fuse = np.concatenate((Y_train_pre1_norm, Y_train_pre2_norm, Y_train_pre3_norm), axis=1)
    Y_val_fuse = np.concatenate((Y_val_pre1_norm, Y_val_pre2_norm, Y_val_pre3_norm), axis=1)

    # binary RF classifier
    # clf = RandomForestClassifier(n_estimators=400, n_jobs=8, min_samples_split=100, min_samples_leaf=50)
    # reg_fuse = RandomForestRegressor(n_estimators=300, n_jobs=8, min_samples_split=200, min_samples_leaf=100)
    reg_fuse = LinearRegression()
    reg_fuse.fit(Y_train_fuse, Y_train_s1a1)

    # with open(save_dir_matFiles + 'nonoverlap_s1a1/fuse/fuse_model_linear.pkl', 'wb') as fid:
    #     pickle.dump(reg_fuse, fid)

    Y_train_predicted_target = reg_fuse.predict(Y_train_fuse)
    Y_val_predicted_target = reg_fuse.predict(Y_val_fuse)

    Y_train_predicted_label = np.zeros((Y_train_pre1.shape[0]))
    Y_val_predicted_label = np.zeros((Y_val_pre1.shape[0]))

    Y_train_predicted_label[Y_train_predicted_target > 0] = 1
    Y_val_predicted_label[Y_val_predicted_target > 0] = 1



    #
    # linear_reg = LinearRegression().fit(Y_train_rf_predicted, Y_train)
    # Y_train_predicted = linear_reg.predict(Y_train_rf_predicted)
    # Y_val_predicted = linear_reg.predict(Y_val_rf_predicted)



    # Y_val_predicted = reg.predict(X_val)
    # Y_train_predicted = reg.predict(X_train)

    # Y_val_label = np.zeros((Y_val.shape))
    # Y_train_label = np.zeros((Y_train.shape))
    #
    #
    # for i in range(Y_val.shape[0]):
    #     if Y_val[i] <= 0:
    #         Y_val_label[i] = -1
    #     else:
    #         Y_val_label[i] = 1
    #
    # for i in range(Y_train.shape[0]):
    #     if Y_train[i] <= 0:
    #         Y_train_label[i] = -1
    #     else:
    #         Y_train_label[i] = 1
    #
    # Y_val_predicted_label = np.zeros((Y_val_predicted.shape))
    # Y_train_predicted_label = np.zeros((Y_train_predicted.shape))
    #
    #
    # for i in range(Y_val_predicted.shape[0]):
    #     if Y_val_predicted[i] <= 0:
    #         Y_val_predicted_label[i] = -1
    #     else:
    #         Y_val_predicted_label[i] = 1
    #
    # for i in range(Y_train.shape[0]):
    #     if Y_train_predicted[i] <= 0:
    #         Y_train_predicted_label[i] = -1
    #     else:
    #         Y_train_predicted_label[i] = 1


    C_train = metrics.confusion_matrix(Y_train_label, Y_train_predicted_label, labels=[0, 1])
    per_class_accuracy_train = np.diag(C_train.astype(np.float32)) / np.sum(C_train.astype(np.float32), axis=1)
    print("train:", per_class_accuracy_train)

    C_val = metrics.confusion_matrix(Y_val_label, Y_val_predicted_label, labels=[0, 1])
    per_class_accuracy_validation = np.diag(C_val.astype(np.float32)) / np.sum(C_val.astype(np.float32), axis=1)
    print("validation:", per_class_accuracy_validation)









            ############       TESTING        ##############
    print("----TEST----")
    X_test_1 = np.concatenate((test_positive1, test_negative1), axis=0)
    X_test_2 = np.concatenate((test_positive2, test_negative2), axis=0)
    test_pos_labels = np.ones((test_positive2.shape[0]))
    test_neg_labels = np.zeros((test_negative2.shape[0]))
    Y_test = np.concatenate((test_pos_labels, test_neg_labels), axis=0)
    print(test_pos_labels.shape, test_neg_labels.shape)


    Y_test_pre_1 = reg_s1a1.predict(X_test_2).reshape(Y_test.shape[0], 1)
    Y_test_pre_2 = reg_s1a2.predict(X_test_2).reshape(Y_test.shape[0], 1)
    Y_test_pre_3 = reg_s2a1.predict(X_test_2).reshape(Y_test.shape[0], 1)

    print(Y_test_pre_1.shape)

    # normalize
    Y_test_pre1_norm = 2 * (Y_test_pre_1 - np.min(Y_test_pre_1)) / (np.max(Y_test_pre_1) - np.min(Y_test_pre_1)) + 1
    Y_test_pre2_norm = 2 * (Y_test_pre_2 - np.min(Y_test_pre_2)) / (np.max(Y_test_pre_2) - np.min(Y_test_pre_2)) + 1
    Y_test_pre3_norm = 2 * (Y_test_pre_3 - np.min(Y_test_pre_3)) / (np.max(Y_test_pre_3) - np.min(Y_test_pre_3)) + 1


    Y_test_fuse = np.concatenate((Y_test_pre1_norm, Y_test_pre2_norm, Y_test_pre3_norm), axis=1)


    Y_test_predicted_2 = reg_fuse.predict(Y_test_fuse)
    Y_test_predicted_label = np.zeros((Y_test_pre_1.shape[0]))

    Y_test_predicted_label[Y_test_predicted_2 > 0] = 1





    # Y_test_predicted = reg.predict(X_test_2)
    #
    # Y_test_binary = np.zeros((Y_test_predicted.shape))
    # for i in range(Y_test_predicted.shape[0]):
    #     if Y_test_predicted[i] <= 0:
    #         Y_test_binary[i] = -1
    #     else:
    #         Y_test_binary[i] = 1

    C_test = metrics.confusion_matrix(Y_test, Y_test_predicted_label, labels=[0, 1])
    per_class_accuracy_test = np.diag(C_test.astype(np.float32)) / np.sum(C_test.astype(np.float32), axis=1)
    print("test acc before double threshold:", per_class_accuracy_test)

    n_p, bins_p, patches_p = plt.hist(x=Y_test_predicted_2[:test_positive2.shape[0]], bins='auto', color='g',
                                alpha=0.7, rwidth=0.85, label="spliced pixels")
    n_n, bins_n, patches_n = plt.hist(x=Y_test_predicted_2[test_positive2.shape[0]:], bins='auto', color='y',
                                alpha=0.7, rwidth=0.85, label="authentic pixels")

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('histogram of testing predicted Y')
    plt.legend()

    plt.savefig(save_dir_matFiles + "nonoverlap_s1a1/fuse/test_histogram_hop2_fuse_linear.png")

    name_loc_prob = pickle.load(open(save_dir_matFiles + 'nonoverlap_s1a1/columbia/name_loc.pkl', 'rb'))

    counts = np.zeros((len(name_loc_prob), 2))

    for k in range(len(name_loc_prob)):
        counts[k][0] = len(name_loc_prob[k]['spliced_loc'])
        counts[k][1] = len(name_loc_prob[k]['authentic_loc'])

    print("total number of spliced and authentic pixels in testing images:", np.sum(counts, axis=0))

    test_spliced_pred = Y_test_predicted_2[:test_positive2.shape[0]]
    test_authen_pred = Y_test_predicted_2[test_positive2.shape[0]:]
    print(test_spliced_pred.shape)
    print(test_authen_pred.shape)

    counts_splice = counts[:,0]
    counts_authen = counts[:,1]


    for k in range(counts.shape[0]):
        splice_pix_num = int(counts[k][0])
        authen_pix_num = int(counts[k][1])
        print(splice_pix_num, authen_pix_num)

        for i in range(splice_pix_num):
            idxi = int(np.sum(counts_splice[:k])) + i
            # print(idxi)
            name_loc_prob[k]['spliced_loc'][i].append(test_spliced_pred[idxi])

        for j in range(authen_pix_num):

            idxj = int(np.sum(counts_authen[:k])) + j
            # print(idxj)
            name_loc_prob[k]['authentic_loc'][j].append(test_authen_pred[idxj])


    with open(save_dir_matFiles + 'nonoverlap_s1a1/fuse/name_loc_prob.pkl', 'wb') as fid:
        pickle.dump(name_loc_prob, fid)


    output_prob_map = np.zeros((len(name_loc_prob), 256, 384))
    gt_map = np.zeros((len(name_loc_prob), 256, 384))

    for k in range(len(name_loc_prob)):
        # name_loc_prob[k] is a dictionary
        splice_pixel_loc = name_loc_prob[k]['spliced_loc']    # list
        authen_pixel_loc = name_loc_prob[k]['authentic_loc']    # list
        for pos_pixel in range(len(splice_pixel_loc)):
            i = splice_pixel_loc[pos_pixel][0]
            j = splice_pixel_loc[pos_pixel][1]
            output_prob_map[k, i, j] = splice_pixel_loc[pos_pixel][2]
            gt_map[k,i,j] = 1

        for neg_pixel in range(len(authen_pixel_loc)):
            i = authen_pixel_loc[neg_pixel][0]
            j = authen_pixel_loc[neg_pixel][1]
            output_prob_map[k, i, j] = authen_pixel_loc[neg_pixel][2]
            gt_map[k,i,j] = -1

    for k in range(len(name_loc_prob)):
        plt.figure(0)
        plt.imshow(output_prob_map[k], cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.savefig(save_dir_matFiles + 'nonoverlap_s1a1/fuse/' + name_loc_prob[k]['test_name'][:-4] + '_output_probmap_hop2_fuse_linear.png')
        plt.close(0)

        # plt.figure(0)
        # plt.imshow(gt_map[k], cmap='coolwarm')
        # plt.colorbar()
        # plt.savefig(save_dir_matFiles + 'columbia/' + name_loc_prob[k]['test_name'][:-4] + 'gt_map.png')
        # plt.close(0)





    # double threshold
    highThreshold = -0.05
    lowThreshold = -0.3

    threshold_output_map, weak_loc = threshold(output_prob_map, lowThreshold=lowThreshold, highThreshold=highThreshold)
    print(threshold_output_map.shape)
    pad_threshold_output_map = np.lib.pad(threshold_output_map, ((0,0), (1, 1), (1,1)), 'constant')

    # hysteresis
    ii = [-1,0,1]
    for k in range(threshold_output_map.shape[0]):
        weak_pixel_loc_i = weak_loc[k][0][0]  # list
        weak_pixel_loc_j = weak_loc[k][0][1]
        print(len(weak_pixel_loc_i))
        for l in range(len(weak_pixel_loc_i)):  # weak pixels number per image
            i = weak_pixel_loc_i[l]
            j = weak_pixel_loc_j[l]

            for m in range(3):
                for n in range(3):
                    # if pad_threshold_output_map[k, i + ii[m]+1, j + ii[n]+1] == -1:   # strong authentic
                    #     threshold_output_map[k,i,j] = -1

                    if pad_threshold_output_map[k, i + ii[m]+1, j + ii[n]+1] == 1:    # strong spliced
                        threshold_output_map[k,i,j] = 1

    for k in range(threshold_output_map.shape[0]):
        M,N = threshold_output_map[k].shape

        leftover_i, leftover_j = np.where((threshold_output_map[k] >= lowThreshold) & (threshold_output_map[k] <= highThreshold))

        threshold_output_map[k,leftover_i,leftover_j] = -1



    # calculate test acc after double thresholding


    Y_test_post = []

    for k in range(len(name_loc_prob)):
        splice_loc_prob = name_loc_prob[k]['spliced_loc']
        authen_loc_prob = name_loc_prob[k]['authentic_loc']

        for pos_pixel in range(len(splice_loc_prob)):
            i = splice_loc_prob[pos_pixel][0]
            j = splice_loc_prob[pos_pixel][1]

            Y_test_post.append(threshold_output_map[k,i,j])

        for neg_pixel in range(len(authen_loc_prob)):
            i = authen_loc_prob[neg_pixel][0]
            j = authen_loc_prob[neg_pixel][1]

            Y_test_post.append(threshold_output_map[k,i,j])

    Y_test_post = np.array(Y_test_post)

    print(Y_test.shape == Y_test_post.shape)

    C_test = metrics.confusion_matrix(Y_test, Y_test_post, labels=[-1, 1])
    per_class_accuracy_test = np.diag(C_test.astype(np.float32)) / np.sum(C_test.astype(np.float32), axis=1)
    print("test acc after post processing:", per_class_accuracy_test)












    _, data = Loss_Guided_KMeans(X_test, Y=None, path='Tree/Kmeantree_regress_v1_all.pkl', train=False, sep_num=2, trial=3, batch_size=100, minS=500,
                       maxN=50, err=0.001, mvth=0.99, maxdepth=50, alpha=1)
    #
    # print(len(data))
    #
    # leaf_node_centroid = []
    # leaf_node_ID = []
    # train_pred = []
    # train_true_target = []
    #
    # for key_1 in data.keys():
    #     if data[key_1]['Data'] != [] and data[key_1]['Target'] != []:
    #         leaf_node_data = data[key_1]['Data']
    #         leaf_node_label = data[key_1]['Target']
    #
    #         reg = Regression_Method(leaf_node_data, leaf_node_label)
    #
    #         data[key_1]['Regressor'] = reg
    #
    #         train_pred.extend(reg.predict(data[key_1]['Data']))
    #
    #         train_true_target.extend(data[key_1]['Target'])
    #
    #         leaf_node_centroid.append(data[key_1]['Centroid'])
    #         leaf_node_ID.append(key_1)
    #
    # train_pred_label = np.zeros((len(train_pred)))
    # train_true_label = np.zeros((len(train_true_target)))
    # for i in range(train_pred_label.shape[0]):
    #     if train_pred[i] <= 0:
    #         train_pred_label[i] = -1
    #     else:
    #         train_pred_label[i] = 1
    #     if train_true_target[i] <= 0:
    #         train_true_label[i] = -1
    #     else:
    #         train_true_label[i] = 1
    #
    # C_train = metrics.confusion_matrix(train_true_label, train_pred_label, labels=[-1, 1])
    # per_class_accuracy_train = np.diag(C_train.astype(np.float32)) / np.sum(C_train.astype(np.float32), axis=1)
    # print("train:", per_class_accuracy_train)
    #
    #
    #
    # test_pred = []
    # for i in range(X_test.shape[0]):
    #     dist = euclidean_distances(X_test[i].reshape(1, -1), leaf_node_centroid)
    #
    #     idx = np.argmin(dist)
    #     ID = leaf_node_ID[idx]
    #
    #     test_pred.append(data[ID]['Regressor'].predict(X_test[i].reshape(1, -1)))
    #
    # test_pred = np.array(test_pred)
    #
    # test_pred_label = np.zeros((test_pred.shape))
    # for i in range(test_pred.shape[0]):
    #     if test_pred[i] <= -0.1:
    #         test_pred_label[i] = -1
    #     else:
    #         test_pred_label[i] = 1
    #
    # C_test = confusion_matrix(Y_test, test_pred_label, labels=[-1, 1])
    #
    #
    # per_class_accuracy_test = np.diag(C_test.astype(np.float32)) / np.sum(C_test.astype(np.float32), axis=1)
    # print("test:", per_class_accuracy_test)
    # #
    # # print("testing mcc score:", mcc(Y_test, test_pred))
    # # print("testing F1 score:", f1(Y_test, test_pred))



    name_loc_prob = pickle.load(open(save_dir_matFiles + 'name_loc_prob.pkl', 'rb'))

    counts = np.zeros((len(name_loc_prob), 2))

    for k in range(len(name_loc_prob)):
        counts[k][0] = len(name_loc_prob[k]['spliced_loc'])
        counts[k][1] = len(name_loc_prob[k]['authentic_loc'])

    print("total number of spliced and authentic pixels in testing images:", np.sum(counts, axis=0))

    # test_pred = np.random.uniform(0, 1, int(np.sum(counts)))
    # test_pred = np.expand_dims(test_pred, axis=1)
    # print("fake:", test_pred.shape)

    test_spliced_pred = test_pred[:test_positive2.shape[0]]
    test_authen_pred = test_pred[test_positive2.shape[0]:]
    print(test_spliced_pred.shape)
    print(test_authen_pred.shape)

    counts_splice = counts[:,0]
    counts_authen = counts[:,1]



    for k in range(counts.shape[0]):
        splice_pix_num = int(counts[k][0])
        authen_pix_num = int(counts[k][1])
        print(splice_pix_num, authen_pix_num)

        for i in range(splice_pix_num):
            idxi = int(np.sum(counts_splice[:k])) + i
            # print(idxi)
            name_loc_prob[k]['spliced_loc'][i].append(test_spliced_pred[idxi][0])

        for j in range(authen_pix_num):

            idxj = int(np.sum(counts_authen[:k])) + j
            # print(idxj)
            name_loc_prob[k]['authentic_loc'][j].append(test_authen_pred[idxj][0])

    with open(save_dir_matFiles + 'name_loc_prob.pkl', 'wb') as fid:
        pickle.dump(name_loc_prob, fid)



    test_pred = np.around(test_pred)


    C_test = confusion_matrix(Y_test, test_pred)


    per_class_accuracy_test = np.diag(C_test.astype(np.float32)) / np.sum(C_test.astype(np.float32), axis=1)
    print(per_class_accuracy_test)

    print("testing mcc score:", mcc(Y_test, test_pred))
    print("testing F1 score:", f1(Y_test, test_pred))

















