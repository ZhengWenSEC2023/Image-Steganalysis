import numpy as np
import time
import pickle
import random
import sklearn.metrics as metrics
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from previous_trial.LAG_yifan import LAG
from previous_trial.LLSR_yifan import LLSR
from sklearn.neighbors import NearestNeighbors

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
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y_target, test_size = 0.1, random_state = 42)

    # train classifier on X_train, Y_train
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=200, max_features='auto', min_samples_leaf=100)
    clf.fit(X_train, Y_train)

    Y_val_predicted = clf.predict(X_val)
    Y_train_predicted = clf.predict(X_train)


    train_loss = metrics.log_loss(Y_train, Y_train_predicted)
    val_loss = metrics.log_loss(Y_val, Y_val_predicted)
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
            # w[j] = X[km.labels_ == j].shape[0] / Y.shape[0]
            t_totalH += ce[j]

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
        # weight = Compute_Weight(kmeans.labels_, num_class)

        for k in range(sep_num):
            t_entropy[i] += Comupte_mse(X['Data'][kmeans.labels_ == k],
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
    return data


def prob_to_label(Y_predicted_prob):
    Y_predict_prob_2 = np.zeros((Y_predicted_prob.shape[0], 2))

    Y_predict_prob_2[:,0] = np.sum(Y_predicted_prob[:, :10], axis=1)
    Y_predict_prob_2[:,1] = np.sum(Y_predicted_prob[:, -10:], axis=1)

    Y_predict_label = np.zeros((Y_predicted_prob.shape[0]))
    for i in range(Y_predicted_prob.shape[0]):
        if Y_predict_prob_2[i,0] > Y_predict_prob_2[i,1]:
            Y_predict_label[i] = 0
        if Y_predict_prob_2[i,0] < Y_predict_prob_2[i,1]:
            Y_predict_label[i] = 1

    # Y_predict_label = np.zeros((Y_predicted_prob.shape[0]))
    # for i in range(Y_predicted_prob.shape[0]):
    #     if Y_predicted_prob[i] > 0:
    #         Y_predict_label[i] = 1
    #     if Y_predicted_prob[i] < 0:
    #         Y_predict_label[i] = 0

    return Y_predict_label



if __name__=='__main__':

    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/'
    print("----Training----")
    # features should be the same
    positive1 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/pos_1_100.npy')
    negative1 = pickle.load(open(save_dir_matFiles + 'nonoverlap_s1a1/neg_1_100.pkl', 'rb'))
    positive2 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/pos_2_100.npy')
    negative2 = pickle.load(open(save_dir_matFiles + 'nonoverlap_s1a1/neg_2_100.pkl', 'rb'))
    print("train shape:", positive2.shape, negative2.shape)



    # X_1 = np.concatenate((positive1, negative1), axis=0)
    X_2 = np.concatenate((positive2, negative2), axis=0)

    # target is different with differnt weights
    pos_target = np.load(save_dir_matFiles + 'nonoverlap_s1a1/pos_target.npy')
    neg_target = pickle.load(open(save_dir_matFiles + 'nonoverlap_s1a1/neg_target.pkl', 'rb'))
    print(np.max(pos_target), np.min(pos_target), np.max(neg_target), np.min(neg_target))

    pos_binary_label = np.ones((positive2.shape[0]), dtype=int)
    neg_binary_label = np.zeros((negative2.shape[0]), dtype=int)
    Y_binary_label = np.concatenate((pos_binary_label, neg_binary_label), axis=0)


    Y_target = np.concatenate((pos_target, neg_target), axis=0)


    # # 10 pseudo-labels
    # pos_label = np.zeros((pos_target.shape)).astype('int')
    # neg_label = np.zeros((neg_target.shape)).astype('int')
    # for k in range(pos_target.shape[0]):
    #     pos_label[k] = np.ceil(pos_target[k]*10)
    # for k in range(neg_target.shape[0]):
    #     neg_label[k] = np.floor(neg_target[k]*10)
    #
    # print(Counter(pos_label).items())
    # print(Counter(neg_label).items())
    #
    # Y_label = np.concatenate((pos_label, neg_label), axis=0)

    # # plot the histogram of pos_label and neg_label
    # pos_n, pos_bins, pos_patches = plt.hist(x=pos_label, bins=np.arange(-10, 10).tolist(), color='g')
    # neg_n, neg_bins, neg_patches = plt.hist(x=neg_label, bins=np.arange(-10, 10).tolist(), color='r')
    # plt.xticks(np.arange(-10, 10, 2))
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('sub-labels histogram')
    # plt.savefig(save_dir_matFiles+'nonoverlap_5by5/hist_digitalized_target.png')


    # read test features, same for different weights
    test_positive1 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/columbia/test_pos_1_40.npy')
    test_negative1 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/columbia/test_neg_1_40.npy')
    test_positive2 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/columbia/test_pos_2_40.npy')
    test_negative2 = np.load(save_dir_matFiles + 'nonoverlap_s1a1/columbia/test_neg_2_40.npy')
    print("test shape:", test_positive2.shape, test_negative2.shape)





    #          ###########       TRAINING       ##############

    # split 80% and 20% training and validation set
    # X_train, X_val, Y_train, Y_val = train_test_split(X_1, Y_target, test_size=0.1, random_state=42)

    X_train, X_val, Y_train, Y_val = train_test_split(X_2, Y_binary_label, test_size=0.2, random_state=42)


    t1 = time.time()
    # feature transformation using LAG + softmax
    ft_trans = LAG(encode='distance', num_clusters=[20, 30], alpha=5, learner=LLSR(onehot=False), neigh=NearestNeighbors(30))

    ft_trans.fit(X_train, Y_train, batch_size=1000, knn=True)

    train_acc, train_knn_label = ft_trans.score(X_train, Y_train, knn=True)
    val_acc, val_knn_label = ft_trans.score(X_val, Y_val, knn=True)

    print("KNN training accuracy:", train_acc)
    print("KNN validation accuracy:", val_acc)
    print("end of training")



#%%
    # X_lag = ft_trans.predict_proba(X_train)
    # print("LAG transformed X: ", X_lag.shape)
    #
    # X_val_lag = ft_trans.predict_proba(X_val)
    # t2 = time.time()
    # print("feature transformation time:", t2-t1)
    #
    #
    # # 1 cluster, 1 classifier
    # clf = RandomForestClassifier(n_estimators=200, n_jobs=8, min_samples_split=100, min_samples_leaf=50)
    # clf.fit(X_lag, Y_train)
    # t3 = time.time()
    # print("train classifier time:", t3-t2)
    #
    # # with open(save_dir_matFiles + 'nonoverlap_s1a1/clf_transform_hop2_1clus.pkl', 'wb') as fid:
    # #     pickle.dump(clf, fid)
    #
    #
    # print("classifier classes:", clf.classes_)
    #
    # # 1 predict method
    # Y_val_predicted = clf.predict(X_val_lag)
    # Y_train_predicted = clf.predict(X_lag)
    #
    # # predicted label
    # Y_val_predicted_label = np.zeros((Y_val_predicted.shape))
    # Y_train_predicted_label = np.zeros((Y_train_predicted.shape))
    #
    # for i in range(Y_val_predicted.shape[0]):
    #     if Y_val_predicted[i] > 0:
    #         Y_val_predicted_label[i] = 1
    #
    # for i in range(Y_train.shape[0]):
    #     if Y_train_predicted[i] > 0:
    #         Y_train_predicted_label[i] = 1
    #
    # # 2 predict probability method
    # Y_train_predicted_prob = clf.predict_proba(X_lag)
    # Y_val_predicted_prob = clf.predict_proba(X_val_lag)
    # Y_train_predicted_label_2 = prob_to_label(Y_train_predicted_prob)    # 0, 1
    # Y_val_predicted_label_2 = prob_to_label(Y_val_predicted_prob)
    #
    # strange = Y_train_predicted_prob[Y_train_predicted_label != Y_train_predicted_label_2]
    #
    #
    # # ground truth
    # Y_val_label = np.zeros((Y_val.shape))
    # Y_train_label = np.zeros((Y_train.shape))
    #
    # for i in range(Y_val.shape[0]):
    #     if Y_val[i] > 0:
    #         Y_val_label[i] = 1
    #
    # for i in range(Y_train.shape[0]):
    #     if Y_train[i] > 0:
    #         Y_train_label[i] = 1

#%%






    # # kmeans clustering in feature space
    # t1 = time.time()
    # n_cluster = 16
    # km = KMeans(n_clusters=n_cluster, random_state=2).fit(X_lag)
    # X_clus_labels = km.labels_
    # unique, count = np.unique(X_clus_labels, return_counts=True)
    # print(dict(label=unique, count=count))
    # t2 = time.time()
    # print("kmeans time:", t2-t1)
    #
    # X_clusters = []
    # Y_clusters = []
    # for k in range(n_cluster):
    #     X_clusters.append(X_lag[X_clus_labels == k])
    #     Y_clusters.append(Y_train[X_clus_labels == k])
    #
    # # regression model in each cluster & average mse
    # reg_clusters = []
    # ave_mses = []
    # Y_clusters_predict_label = []
    # Y_clusters_label = []
    #
    # # with open(save_dir_matFiles + 'nonoverlap_s1a1/kmeans/model_32.pkl', 'rb') as fid:
    # #     reg_clusters = pickle.load(fid)
    #
    #
    # for k in range(n_cluster):
    #     X_cluster = X_clusters[k]
    #     Y_cluster = Y_clusters[k]
    #
    #     # count the number of splice and authentic pixels in each cluster
    #     p = len(Y_cluster[Y_cluster > 0])
    #     n = len(Y_cluster[Y_cluster < 0])
    #     print("cluster {}, spliced: {}, authentic: {}".format(k, p/(p+n), n/(p+n)))
    #
    #
    #
    #     reg = RandomForestClassifier(n_estimators=100, n_jobs=8,
    #                                 min_samples_split=200, min_samples_leaf=100)
    #
    #     # reg = LinearRegression()
    #
    #     reg.fit(X_cluster, Y_cluster)
    #     reg_clusters.append(reg)
    #     # reg = reg_clusters[k]
    #
    #     Y_cluster_predict_prob = reg.predict_proba(X_cluster)
    #
    #     Y_cluster_predict_label = prob_to_label(Y_cluster_predict_prob)
    #
    #     Y_clusters_predict_label.extend(Y_cluster_predict_label)
    #
    #     Y_cluster_label = np.zeros((Y_cluster.shape[0]))
    #     Y_cluster_label[Y_cluster > 0] = 1
    #     Y_clusters_label.extend(Y_cluster_label)
    #
    #     # each cluster accuracy
    #     C_cluster = metrics.confusion_matrix(Y_cluster_label, Y_cluster_predict_label, labels=[0, 1])
    #     per_class_accuracy_cluster = np.diag(C_cluster.astype(np.float32)) / np.sum(C_cluster.astype(np.float32), axis=1)
    #     print("cluster {} accuracy:".format(k), per_class_accuracy_cluster)
    #
    #     # mse = mean_squared_error(Y_cluster, Y_cluster_predict)
    #     # ave_mse = mse / X_cluster.shape[0]
    #
    #     # ave_mses.append(ave_mse)
    #
    # flatten = lambda l: [item for sublist in l for item in sublist]
    #
    #
    #
    #
    # #
    # # with open(save_dir_matFiles + 'nonoverlap_s1a1/kmeans/model_16_svr.pkl', 'wb') as fid:
    # #     pickle.dump(reg_clusters, fid)
    #
    # t3 = time.time()
    # print("train model time:", t3-t2)
    #
    # # validation set
    # Y_val_clusters_predict = []
    # Y_val_clusters = []
    #
    # X_val_kmlabels = km.predict(X_val_lag)
    # for k in range(n_cluster):
    #     Y_val_cluster_predict = reg_clusters[k].predict_proba(X_val_lag[X_val_kmlabels == k])
    #     Y_val_cluster_predict_label = prob_to_label(Y_val_cluster_predict)
    #     Y_val_clusters_predict.extend(Y_val_cluster_predict_label)
    #
    #     Y_val_cluster = Y_val[X_val_kmlabels == k]
    #     Y_val_cluster_label = np.zeros((Y_val_cluster.shape[0]))
    #     Y_val_cluster_label[Y_val_cluster > 0] = 1
    #     Y_val_clusters.extend(Y_val_cluster_label)






    # C_train = metrics.confusion_matrix(Y_clusters_label, Y_clusters_predict_label, labels=[0, 1])
    # per_class_accuracy_train = np.diag(C_train.astype(np.float32)) / np.sum(C_train.astype(np.float32), axis=1)
    # print("total train accuracy:", per_class_accuracy_train)
    #
    # C_val = metrics.confusion_matrix(Y_val_clusters, Y_val_clusters_predict, labels=[0, 1])
    # per_class_accuracy_validation = np.diag(C_val.astype(np.float32)) / np.sum(C_val.astype(np.float32), axis=1)
    # print("total validation accuracy:", per_class_accuracy_validation)








    # C_train_2 = metrics.confusion_matrix(Y_train_label, Y_train_predicted_label_2, labels=[0, 1])
    # per_class_accuracy_train = np.diag(C_train_2.astype(np.float32)) / np.sum(C_train_2.astype(np.float32), axis=1)
    # print("total train accuracy:", per_class_accuracy_train)
    #
    # C_val_2 = metrics.confusion_matrix(Y_val_label, Y_val_predicted_label_2, labels=[0, 1])
    # per_class_accuracy_validation = np.diag(C_val_2.astype(np.float32)) / np.sum(C_val_2.astype(np.float32), axis=1)
    # print("total validation accuracy:", per_class_accuracy_validation)






#%%


            ############       TESTING        ##############
    print("----TEST----")
    X_test_1 = np.concatenate((test_positive1, test_negative1), axis=0)
    X_test_2 = np.concatenate((test_positive2, test_negative2), axis=0)
    test_pos_labels = np.ones((test_positive2.shape[0]), dtype=int)
    test_neg_labels = np.zeros((test_negative2.shape[0]),dtype=int)
    Y_test = np.concatenate((test_pos_labels, test_neg_labels), axis=0)
    print(test_pos_labels.shape, test_neg_labels.shape)

    # X_test_lag = ft_trans.predict_proba(X_test_2)
    test_acc, test_knn_label = ft_trans.score(X_test_2, Y_test, knn=True)
    print("KNN testing accuracy:", test_acc)


    #1 predict method
    Y_test_predict = clf.predict(X_test_lag)   # 20
    Y_test_predict_label = np.zeros((Y_test_predict.shape[0]))
    for i in range(Y_test_predict.shape[0]):
        if Y_test_predict[i] > 0:
            Y_test_predict_label[i] = 1


    # 2 predict proba method
    Y_test_predict_prob = clf.predict_proba(X_test_lag)
    Y_test_predict_label_2 = prob_to_label(Y_test_predict_prob)





#%%
    # X_test_kmlabel = km.predict(X_test_lag)
    #
    # Y_test_clusters_predict = []
    # Y_test_clusters = []
    #
    # Y_test_splice = []
    # Y_test_authentic = []
    #
    # for k in range(n_cluster):
    #     X_test_cluster = X_test_lag[X_test_kmlabel == k]
    #
    #     if X_test_cluster.shape[0] != 0:
    #         Y_test_cluster = Y_test[X_test_kmlabel == k]
    #         Y_test_cluster_predict = reg_clusters[k].predict_proba(X_test_cluster)
    #         Y_test_cluster_predict_label = prob_to_label(Y_test_cluster_predict)
    #
    #         # predict label
    #         Y_test_clusters_predict.extend(Y_test_cluster_predict_label)
    #         # true label
    #         Y_test_clusters.extend(Y_test_cluster)
    #
    #         # spliced pixels probability
    #         Y_test_cluster_splice_prob = np.sum(Y_test_cluster_predict[:,-10:], axis=1)
    #         Y_test_splice.append(Y_test_cluster_splice_prob[Y_test_cluster == 1])
    #
    #         # authentic pixels probability
    #         Y_test_authentic.append(Y_test_cluster_splice_prob[Y_test_cluster == 0])
    #
    #
    #
    # Y_test_splice = flatten(Y_test_splice)
    # Y_test_authentic = flatten(Y_test_authentic)
    # print("length of testing spliced and authentic pixels:", len(Y_test_splice), len(Y_test_authentic))


    # Y_test_predict_prob_20 = clf.predict_proba(X_test_2)    # 220000, 20
    #
    # Y_test_predicted = np.zeros((Y_test.shape[0],2))
    # Y_test_predicted[:,0] = -1*(np.sum(Y_test_predict_prob_20[:, :10], axis=1))
    # Y_test_predicted[:,1] = np.sum(Y_test_predict_prob_20[:, -10:], axis=1)
    #
    # Y_test_predict_label = np.zeros((Y_test.shape[0]))
    # for i in range(Y_test.shape[0]):
    #     if np.abs(Y_test_predicted[i,0]) > np.abs(Y_test_predicted[i,1]):
    #         Y_test_predict_label[i] = 0
    #     if np.abs(Y_test_predicted[i,0]) < np.abs(Y_test_predicted[i,1]):
    #         Y_test_predict_label[i] = 1

#%%

    # C_test = metrics.confusion_matrix(Y_test_clusters, Y_test_clusters_predict, labels=[0, 1])
    # per_class_accuracy_test = np.diag(C_test.astype(np.float32)) / np.sum(C_test.astype(np.float32), axis=1)
    # print("test acc before double threshold:", per_class_accuracy_test)

    C_test_2 = metrics.confusion_matrix(Y_test, Y_test_predict_label_2, labels=[0, 1])
    per_class_accuracy_test = np.diag(C_test_2.astype(np.float32)) / np.sum(C_test_2.astype(np.float32), axis=1)
    print("test acc before double threshold:", per_class_accuracy_test)

    Y_test_prob_spliced = np.sum(Y_test_predict_prob[:,-10:], axis=1)
    Y_test_splice = Y_test_prob_spliced[:test_positive2.shape[0]]
    Y_test_authentic = Y_test_prob_spliced[test_positive2.shape[0]:]


    n_p, bins_p, patches_p = plt.hist(x=Y_test_splice, bins='auto', color='b',
                                alpha=0.7, rwidth=0.85, label="spliced pixels")
    n_n, bins_n, patches_n = plt.hist(x=Y_test_authentic, bins='auto', color='y',
                                alpha=0.7, rwidth=0.85, label="authentic pixels")

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('histogram of testing predicted Y')
    plt.legend()

    plt.savefig(save_dir_matFiles + "nonoverlap_s1a1/kmeans/test_hist_hop2_fttrans_1clus.png")

    name_loc_prob = pickle.load(open(save_dir_matFiles + 'nonoverlap_s1a1/columbia/name_loc.pkl', 'rb'))

    counts = np.zeros((len(name_loc_prob), 2))

    for k in range(len(name_loc_prob)):
        counts[k][0] = len(name_loc_prob[k]['spliced_loc'])
        counts[k][1] = len(name_loc_prob[k]['authentic_loc'])

    print("total number of spliced and authentic pixels in testing images:", np.sum(counts, axis=0))

    test_spliced_pred = np.array(Y_test_splice)
    test_authen_pred = np.array(Y_test_authentic)

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


    with open(save_dir_matFiles + 'nonoverlap_s1a1/kmeans/name_loc_prob_fttrans_1clus.pkl', 'wb') as fid:
        pickle.dump(name_loc_prob, fid)


    output_prob_map = np.zeros((len(name_loc_prob), 256, 384)) + 0.5
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
        plt.imshow(output_prob_map[k], cmap='coolwarm', vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(save_dir_matFiles + 'nonoverlap_s1a1/kmeans/' + name_loc_prob[k]['test_name'][:-4] + '_output_probmap_hop2_fttrans_1clus.png')
        plt.close(0)

        # plt.figure(0)
        # plt.imshow(gt_map[k], cmap='coolwarm')
        # plt.colorbar()
        # plt.savefig(save_dir_matFiles + 'columbia/' + name_loc_prob[k]['test_name'][:-4] + 'gt_map.png')

