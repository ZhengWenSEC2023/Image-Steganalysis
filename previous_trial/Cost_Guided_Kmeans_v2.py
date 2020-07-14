import os
import sys
import numpy as np
import time
import pickle
import scipy
import sklearn
import math
import random
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


def Regression_Method(X, Y):

    reg = RandomForestRegressor(max_depth=None, n_estimators=50, random_state=0, min_samples_split= 4, min_samples_leaf=2)
    # reg = LinearRegression()
    # reg = LogisticRegression(solver='saga', multi_class='ovr', n_jobs=12)

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

    print("       <Debug Info>        Entropy: %s" % str(H_val))
    print("       <Debug Info>        Weight: %s" % str(w))
    return gH_val, gH_train



################################# Cross Entropy #################################
# used when computing entropy in <Multi_Trial>
def Compute_Weight(Y):
    weight = np.zeros(np.unique(Y).shape[0])     # 2,
    for i in range(0, weight.shape[0]):
        if (Y[Y == i].shape[0]) == 0:
            weight[i] = 0
        else:
            weight[i] = (float)(Y[Y == i].shape[0])
    weight /= np.sum(weight)
    return weight


# latest cross entropy method
def Comupte_Cross_Entropy(X, Y, num_class):
    samp_num = Y.size
    if np.unique(Y).shape[0] == 1:  # already pure
        return 0,0
    # if X.shape[0] < num_bin:     # sample num in cluster is too small!
    #     return -1

    # split 80% and 20% training and validation set
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    # train classifier on X_train, Y_train
    clf = RandomForestClassifier(n_estimators=50, class_weight='balanced', min_samples_split=2, max_features='auto',
                           bootstrap=True, max_depth=5, min_samples_leaf=2)
    clf.fit(X_train, Y_train)

    Y_val_predicted = clf.predict_proba(X_val)
    Y_train_predicted = clf.predict_proba(X_train)

    # kmeans = KMeans(n_clusters=num_bin, random_state=0).fit(X)
    #     # prob = np.zeros((num_bin, num_class))      # 32, 2
    #     # for i in range(num_bin):
    #     #     idx = (kmeans.labels_ == i)
    #     #     tmp = Y[idx]
    #     #     for j in range(num_class):
    #     #         prob[i, j] = (float)(tmp[tmp == j].shape[0]) / ((float)(Y[Y == j].shape[0]) + 1e-5)
    #     # prob = prob / (np.sum(prob, axis=1).reshape(-1, 1) + 1e-5)
    #     #
    #     # # true_indicator = np.zeros((samp_num, num_class))
    #     # # true_indicator[np.arange(samp_num), Y] = 1
    #     # true_indicator = Y
    #     # probab = prob[kmeans.labels_]

    return LL(Y_val, Y_val_predicted), LL(Y_train, Y_train_predicted) # / math.log(num_class)


################################# Init For Root Node #################################
# init kmeans centroid with center of samples from each label, then do kmeans
def Init_By_Class(X, Y, num_class, sep_num, trial):
    init_centroid = []
    for i in range(np.unique(Y).shape[0]):
        Y = Y.reshape(-1)
        init_centroid.append(np.mean(X[Y == i], axis=0).reshape(-1))
    tmpH = 10
    init_centroid = np.array(init_centroid)

    for i in range(trial):
        t = np.arange(0, np.unique(Y).shape[0]).tolist()    # 0,1
        tmp_idx = np.array(random.sample(t, len(t)))
        km = KMeans(n_clusters=sep_num, init=init_centroid[tmp_idx[0:sep_num]]).fit(X)

        ce = np.zeros((sep_num))
        w = np.zeros((sep_num))
        t_totalH = 0.0
        for j in range(sep_num):  # 0, 1
            ce[j] = Comupte_Cross_Entropy(X[km.labels_ == j], Y[km.labels_ == j], num_class)[0]
            w[j] = X[km.labels_ == j].shape[0] / Y.shape[0]
            t_totalH += w[j]*ce[j]

        if tmpH > t_totalH:
            kmeans = km
            tmpH = t_totalH

    data = []
    data.append({'Data': [],
                 'Label': [],
                 'Centroid': np.mean(X, axis=1),
                 'H_val': Comupte_Cross_Entropy(X, Y, num_class)[0],
                 'H_train': Comupte_Cross_Entropy(X, Y, num_class)[1],
                 'ID': str(-1)})
    H = []
    Hidx = [1]
    for i in range(sep_num):    # 0, 1
        data.append({'Data': X[kmeans.labels_ == i],
                     'Label': Y[kmeans.labels_ == i],
                     'Centroid': kmeans.cluster_centers_[i],
                     'H_val': Comupte_Cross_Entropy(X[kmeans.labels_ == i], Y[kmeans.labels_ == i], num_class)[0],
                     'H_train': Comupte_Cross_Entropy(X[kmeans.labels_ == i], Y[kmeans.labels_ == i], num_class)[1],
                     'ID': str(i)})
        H.append(data[i]['H_val'])        # root node ce, child node 1 ce, child node 2 ce
        Hidx.append(Hidx[-1] + 1)     # 1, 2, 3

    X0 = data[1]['Data']
    X1 = data[2]['Data']
    Y0 = data[1]['Label']
    Y1 = data[2]['Label']

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


# init whole data as one cluster
def Init_As_Whole(X, Y, num_class):
    data = [{'Data': X,
             'Label': Y,
             'Centroid': np.mean(X, axis=0),
             'H_val': Comupte_Cross_Entropy(X, Y, num_class)[0],
             'H_train': Comupte_Cross_Entropy(X, Y, num_class)[1],
             'ID': '0'}]
    H = [data[0]['H_val']]
    Hidx = [0, 1]
    return data, H, Hidx


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
        weight = Compute_Weight(kmeans.labels_)

        for k in range(sep_num):
            t_entropy[i] += weight[k] * Comupte_Cross_Entropy(X['Data'][kmeans.labels_ == k],
                                                              X['Label'][kmeans.labels_ == k], num_class)[0]
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
        subX.append({'Data': X['Data'][idx], 'Label': X['Label'][idx], 'Centroid': center[i],
                     'H_val': Comupte_Cross_Entropy(X['Data'][idx], X['Label'][idx], num_class)[0],
                     'H_train': Comupte_Cross_Entropy(X['Data'][idx], X['Label'][idx], num_class)[1],
                     'ID': X['ID'] + str(i)})
    return subX


def Leaf_Node_Regression(data, Hidx, num_class):
    for i in range(len(Hidx) - 1):
        data[Hidx[i]]['Regressor'] = Regression_Method(data[Hidx[i]]['Data'], data[Hidx[i]]['Label'], num_class)
        data[Hidx[i]]['Data'] = []  # no need to store raw data any more
        data[Hidx[i]]['Label'] = []
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
    num_class = np.unique(Y).shape[0]
    data, H, Hidx = Init_By_Class(X, Y, num_class, sep_num, trial)    # data: list 里面多个dict, 每一个node都是一个dict
    rootSampNum = Y.shape[0]    # total number of samples in root
    global_H_val = []
    global_H_train = []

    X, Y = [], []
    N, myiter = 1, 1
    print("\n       <Info>        Start iteration")
    print("       <Info>        Iter %s" % (str(0)))
    global_H_val.append(Compute_GlobalH(data, rootSampNum, Hidx)[0])
    global_H_train.append(Compute_GlobalH(data, rootSampNum, Hidx)[1])

    while N < maxN:
        print("       <Info>        Iter %s" % (str(myiter)))
        idx = Select_Next_Split(data, Hidx, alpha)
        print("           <Debug>       Hidx:", Hidx)
        print("           <Debug>       which idx to split:", idx)
        print("           <Debug>       Global H val:", global_H_val)
        print("           <Debug>       Global H train:", global_H_train)

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
            data[Hidx[idx]]['Label'] = []
            data += subX    # 加入两个child node {} 信息
            H.pop(idx)
            Hidx.pop(idx)
            N -= 1
            for d in subX:
                H.append(d['H_val'])
                Hidx.append(Hidx[-1] + 1)
                N += 1
            myiter += 1
            global_H_val.append(Compute_GlobalH(data, rootSampNum, Hidx)[0])
            global_H_train.append(Compute_GlobalH(data, rootSampNum, Hidx)[1])

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

    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/'

    positive2 = np.load(save_dir_matFiles + 'pos_2_100.npy')[:5000]
    negative2 = pickle.load(open(save_dir_matFiles + 'neg_2_100.pkl', 'rb'))[:5000]

    test_positive2 = np.load(save_dir_matFiles + 'test_pos_2_100.npy')[:5000]
    test_negative2 = np.load(save_dir_matFiles + 'test_neg_2_100.npy')[:5000]

    X = np.concatenate((positive2, negative2), axis=0)
    pos_labels = np.ones((positive2.shape[0],))
    neg_labels = np.zeros((negative2.shape[0],))
    Y = np.concatenate((pos_labels, neg_labels), axis=0)

             ###########       TRAINING       ##############
    # X, data = Loss_Guided_KMeans(X, Y, path='Tree/Kmeantree_multinode_all.pkl', train=True, sep_num=2, trial=3, batch_size=1000, minS=1000, maxN=50,
    #            err=0.005, mvth=0.99, maxdepth=50, alpha=1)


            ############       TESTING        ##############

    X_test = np.concatenate((test_positive2, test_negative2), axis=0)
    test_pos_labels = np.ones((test_positive2.shape[0]))
    test_neg_labels = np.zeros((test_negative2.shape[0]))
    Y_test = np.concatenate((test_pos_labels, test_neg_labels), axis=0)
    print(test_pos_labels.shape, test_neg_labels.shape)


    _, data = Loss_Guided_KMeans(X_test, Y=None, path='Tree/Kmeantree_multinode_all.pkl', train=False, sep_num=2, trial=3, batch_size=1000, minS=1000,
                       maxN=50, err=0.005, mvth=0.99, maxdepth=50, alpha=1)

    print(len(data))

    leaf_node_centroid = []
    leaf_node_ID = []
    train_pred = []
    train_true = []

    for key_1 in data.keys():
        if data[key_1]['Data'] != [] and data[key_1]['Label'] != []:
            leaf_node_data = data[key_1]['Data']
            leaf_node_label = data[key_1]['Label']

            reg = Regression_Method(leaf_node_data, leaf_node_label)

            data[key_1]['Regressor'] = reg

            train_pred.extend(reg.predict(data[key_1]['Data']))

            train_true.extend(data[key_1]['Label'])

            leaf_node_centroid.append(data[key_1]['Centroid'])
            leaf_node_ID.append(key_1)


    # train_true = np.array(train_true)
    # train_pred = np.array(train_pred)
    # train_pred = np.around(train_pred)
    # print(train_true.shape)
    # print(train_pred.shape)
    #
    # C_train = confusion_matrix(train_true, train_pred)
    # per_class_accuracy_train = np.diag(C_train.astype(np.float32)) / np.sum(C_train.astype(np.float32), axis=1)
    # print(per_class_accuracy_train)
    #
    # print("training mcc score:", mcc(train_true, train_pred))
    # print("training F1 score:", f1(train_true, train_pred))

    test_pred = []
    for i in range(X_test.shape[0]):
        dist = euclidean_distances(X_test[i].reshape(1, -1), leaf_node_centroid)

        idx = np.argmin(dist)
        ID = leaf_node_ID[idx]

        test_pred.append(data[ID]['Regressor'].predict(X_test[i].reshape(1, -1)))

    test_pred = np.array(test_pred)
    # test_pred = np.around(test_pred)
    #
    #
    # C_test = confusion_matrix(Y_test, test_pred)
    #
    #
    # per_class_accuracy_test = np.diag(C_test.astype(np.float32)) / np.sum(C_test.astype(np.float32), axis=1)
    # print(per_class_accuracy_test)
    #
    # print("testing mcc score:", mcc(Y_test, test_pred))
    # print("testing F1 score:", f1(Y_test, test_pred))



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

















