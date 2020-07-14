# 2020.04.23
# Hierachical Kmeans
# author: yifan
# modified by Yao Zhu


import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.preprocessing as pre_processing

# from mylearner import myLearner


# LBG initialization
def Init_LBG_plus(X, sep_num=2):
    c1 = np.mean(X, axis=0).reshape(1, -1)
    st = np.std(X, axis=0).reshape(1, -1)
    c2 = c1 + st
    new_centroids = np.concatenate((c1, c2), axis=0)
    return new_centroids


def Init_LBG_minus(X, sep_num=2):
    c1 = np.mean(X, axis=0).reshape(1, -1)
    st = np.std(X, axis=0).reshape(1, -1)
    c2 = c1 - st
    new_centroids = np.concatenate((c1, c2), axis=0)
    return new_centroids


def Init_Farthest(X, sep_num=2, pairs=3):
    # only support sep_num=2 now
    centroid = np.mean(X, axis=0, keepdims=1)
    farthest_set, _ = find_farthest(X, centroid, number=100)

    new_centroids = []

    # first pair
    seed0_idx = np.random.choice(100)
    seed0 = farthest_set[[seed0_idx]]
    farthest_set = np.delete(farthest_set, seed0_idx, axis=0)  # delete seed0 from candidates set
    seed1, seed1_idx = find_farthest(farthest_set, seed0, number=1)  # find farthest seed from seed0
    farthest_set = np.delete(farthest_set, seed1_idx, axis=0)  # delete seed1 from candidates set
    new_centroids.append(np.concatenate((seed0, seed1), axis=0))

    if pairs > 1:
        # second pair seed 0, is the argmax(min(dist2pair1))
        dist2seed0 = cal_dist(farthest_set, seed0).reshape(1, -1)
        dist2seed1 = cal_dist(farthest_set, seed1).reshape(1, -1)
        dist2pair1 = np.concatenate((dist2seed0, dist2seed1), axis=0)  # (2, xxx)
        del dist2seed0, dist2seed1

        dist2pair1_min = np.min(dist2pair1, axis=0)
        seed2_idx = np.argmax(dist2pair1_min)
        seed2 = farthest_set[[seed2_idx]]  # seed 0 for pair 2
        farthest_set = np.delete(farthest_set, seed2_idx, axis=0)
        seed3, seed3_idx = find_farthest(farthest_set, seed2, number=1)  # seed 1 for pair 2
        farthest_set = np.delete(farthest_set, seed3_idx, axis=0)
        new_centroids.append(np.concatenate((seed2, seed3), axis=0))

        if pairs > 2:
            # third pair, check cosine similarity
            direction1 = seed1 - seed0
            direction2 = seed3 - seed2
            del seed0, seed1, seed2, seed3
            pair_idx = np.array(list(itertools.combinations(np.arange(farthest_set.shape[0]), 2)))
            rest_pairs = farthest_set[pair_idx[:, 0]] - farthest_set[pair_idx[:, 1]]  # rest pair vectors
            cos_sim_1 = cosine_similarity(rest_pairs, direction1).reshape(1, -1)
            cos_sim_2 = cosine_similarity(rest_pairs, direction2).reshape(1, -1)
            cos_sim = np.concatenate((cos_sim_1, cos_sim_2), axis=0)
            cos_sim_max = np.max(np.abs(cos_sim), axis=0)
            pair3_idx = np.argmin(cos_sim_max)
            new_centroids.append(
                np.concatenate((farthest_set[[pair_idx[pair3_idx][0]]], farthest_set[[pair_idx[pair3_idx][1]]]),
                               axis=0))

            farthest_set = np.delete(farthest_set, [pair_idx[pair3_idx][0], pair_idx[pair3_idx][1]], axis=0)

    pair1 = (new_centroids[0][1] - new_centroids[0][0]).reshape(1, -1)
    pair2 = (new_centroids[1][1] - new_centroids[1][0]).reshape(1, -1)
    pair3 = (new_centroids[2][1] - new_centroids[2][0]).reshape(1, -1)

    tmp = [pair1, pair2, pair3]
    for k in range(2, pairs):
        pair_idx = np.array(list(itertools.combinations(np.arange(farthest_set.shape[0]), 2)))
        rest_pairs = farthest_set[pair_idx[:, 0]] - farthest_set[pair_idx[:, 1]]  # rest pair vectors
        cos_sim = []
        for i in range(len(tmp)):
            cos_sim_i = cosine_similarity(rest_pairs, tmp[i]).reshape(-1)
            cos_sim.append(cos_sim_i)
        cos_sim = np.array(cos_sim)
        cos_sim_max = np.max(np.abs(cos_sim), axis=0)
        pairn_idx = np.argmin(cos_sim_max)
        print(pair_idx[pairn_idx])

        new_centroids.append(
            np.concatenate((farthest_set[[pair_idx[pairn_idx][0]]], farthest_set[[pair_idx[pairn_idx][1]]]),
                           axis=0))
        farthest_set = np.delete(farthest_set, [pair_idx[pairn_idx][0], pair_idx[pairn_idx][1]], axis=0)

        pairn = (new_centroids[-1][1] - new_centroids[-1][0]).reshape(1, -1)
        tmp.append(pairn)

    return new_centroids


def cal_dist(samples, target):
    distance = np.sum(np.power((samples - target), 2), axis=1)
    return distance


def find_farthest(samples, target, number=1):
    dist = cal_dist(samples, target)
    rank_dist = np.argsort(-1 * dist)
    farthest_set = samples[[rank_dist[:number]]]
    return farthest_set, rank_dist[:number]


def Loss_function(kmean, X, Y, sse_parent, mse_parent):
    label = kmean.labels_

    # SSE of two child nodes
    X1 = X[label == 0]
    X2 = X[label == 1]

    # Balanceness
    min_num = np.min((X1.shape[0], X2.shape[0]), axis=0)
    max_num = np.max((X1.shape[0], X2.shape[0]), axis=0)
    L_balance = 1 - min_num / max_num

    X1_m = np.mean(X1, axis=0)
    X2_m = np.mean(X2, axis=0)

    X1_pred = np.ones((X1.shape)) * X1_m
    X2_pred = np.ones((X2.shape)) * X2_m

    sse_c1 = mean_squared_error(X1, X1_pred) * len(X1)  # inertia of c1
    sse_c2 = mean_squared_error(X2, X2_pred) * len(X2)  # inertia of c2

    L_sse = (sse_c1 + sse_c2) / sse_parent  # 1-x

    # MSE of two child nodes: train Least square regression model
    Y1 = Y[label == 0]
    Y2 = Y[label == 1]
    # X1_b = np.vstack([X1.T, np.ones((len(X1)))]).T
    # X2_b = np.vstack([X2.T, np.ones((len(X2)))]).T
    # X_b = np.vstack([X.T, np.ones((len(X)))]).T
    #
    # a1 = np.linalg.lstsq(X1_b, Y1, rcond=None)[0]       # slow!
    # a2 = np.linalg.lstsq(X2_b, Y2, rcond=None)[0]
    # a = np.linalg.lstsq(X_b, Y, rcond=None)[0]

    # pred_Y1 = np.matmul(X1_b, a1)
    # pred_Y2 = np.matmul(X2_b, a2)
    # pred_Y = np.matmul(X_b, a)


    prob_Y1_1 = np.sum(Y1 == 1) / len(Y1)
    prob_Y1_0 = np.sum(Y1 == 0) / len(Y1)
    prob_Y2_1 = np.sum(Y2 == 1) / len(Y2)
    prob_Y2_0 = np.sum(Y2 == 0) / len(Y2)
    log_prob_Y1_1 = np.log(prob_Y1_1)
    log_prob_Y1_0 = np.log(prob_Y1_0)
    log_prob_Y2_1 = np.log(prob_Y2_1)
    log_prob_Y2_0 = np.log(prob_Y2_0)

    log_prob_Y1 = np.array([[log_prob_Y1_0], [log_prob_Y1_1]])
    log_prob_Y2 = np.array([[log_prob_Y2_0], [log_prob_Y2_1]])

    onehot_fit = pre_processing.OneHotEncoder()
    Y1 = onehot_fit.fit_transform(np.reshape(Y1, (Y1.shape[0], 1))).toarray()
    Y2 = onehot_fit.fit_transform(np.reshape(Y2, (Y2.shape[0], 1))).toarray()

    reg1 = LinearRegression().fit(X1, Y1)
    reg2 = LinearRegression().fit(X2, Y2)

    pred_Y1 = reg1.predict(X1)
    pred_Y2 = reg2.predict(X2)

    pred_Y1[pred_Y1 > 1] = 1
    pred_Y1[pred_Y1 < 0] = 0
    pred_Y2[pred_Y2 > 1] = 1
    pred_Y2[pred_Y2 < 0] = 0

    L_ce = (np.sum(Y1 * pred_Y1) / len(Y1) + np.sum(Y2 * pred_Y2) / len(Y2))

    return L_balance + L_sse + L_ce


def test_predict(test_scaled, top_leaf_centroid_, regs, k):
    dist = euclidean_distances(top_leaf_centroid_,
                               test_scaled[k].reshape(1, test_scaled.shape[-1]))  # euclidean distance to centroids 11-D

    idx = int(np.argmin(dist))

    if type(regs[idx]) is np.float64:
        pred = regs[idx]

        pred = np.array([pred])
        print("！！！", k, pred, pred.dtype, type(pred))

    else:
        pred = regs[idx].predict(test_scaled[k].reshape(1, test_scaled.shape[-1]))

    return pred


class HierNode():
    def __init__(self, num_cluster, metric, isleaf=False, id='R'):
        # self.learner = learner
        self.kmeans = []
        self.num_cluster = num_cluster
        self.metric = metric
        self.isleaf = isleaf
        self.id = id
        self.centroid = []
        self.source_mse_parent = 0.0
        self.source_mse_child = np.zeros((num_cluster))

        self.gtvar_parent = 0.0
        self.gtvar_child = np.zeros((num_cluster))

    def metric_(self, X, Y):
        # if 'func' in self.metric.keys():
        #     return self.metric['func'](X, self.metric)
        if X.shape[0] < self.metric['min_num_sample']:
            return True
        if len(Y[Y > 0]) / len(Y) > self.metric['purity'] or len(Y[Y < 0]) / len(Y) > self.metric['purity']:
            return True

        return False

    def multi_trial(self, X, Y):  # try different initialization method on kmeans
        km_candidate = []
        loss = []
        inertia = []

        # parent SSE and MSE
        X_m = np.mean(X, axis=0)
        X_pred = np.ones((X.shape)) * X_m
        sse = mean_squared_error(X, X_pred) * len(X)

        reg = LinearRegression().fit(X, Y)
        pred_Y = reg.predict(X)
        mse = mean_squared_error(Y, pred_Y)

        # 1. LBG_plus
        initial_centroids = Init_LBG_plus(X)
        km_LBG_plus = MiniBatchKMeans(n_clusters=2, batch_size=1000, init=initial_centroids, n_init=1).fit(X)
        km_candidate.append(km_LBG_plus)
        inertia.append(km_LBG_plus.inertia_)
        loss.append(Loss_function(km_LBG_plus, X, Y, sse, mse))

        # 2. farthest pair
        pair_num = 10
        initial_centroids = Init_Farthest(X, sep_num=2, pairs=pair_num)
        for k in range(pair_num):
            km_farthest = MiniBatchKMeans(n_clusters=2, batch_size=1000, init=initial_centroids[k], n_init=1).fit(X)
            km_candidate.append(km_farthest)
            loss.append(Loss_function(km_farthest, X, Y, sse, mse))
            inertia.append(km_farthest.inertia_)

        # 3. k-means++
        km_plus = MiniBatchKMeans(n_clusters=2, batch_size=1000, init='k-means++', n_init=20).fit(X)
        km_candidate.append(km_plus)
        inertia.append(km_plus.inertia_)
        loss.append(Loss_function(km_plus, X, Y, sse, mse))

        # 4. random sample
        km_random = MiniBatchKMeans(n_clusters=2, batch_size=1000, init='random', n_init=20).fit(X)
        km_candidate.append(km_random)
        inertia.append(km_random.inertia_)
        loss.append(Loss_function(km_random, X, Y, sse, mse))

        # find the split with smallest loss function (smallest penalty)
        idx = int(np.argmin(loss))
        self.kmeans = km_candidate[idx]

        print(self.kmeans)

    def fit(self, X, Y):

        self.multi_trial(X, Y)
        # self.kmeans.fit(X)     # try different initialization method on kmeans

    def predict(self, X):

        return self.kmeans.predict(X)

    def source_mse_decrease(self, X, Y):
        clus_label = self.predict(X)

        self.source_mse_parent = np.mean(np.var(X, axis=0))

        self.gtvar_parent = np.var(Y) * len(Y)  # total variance, not average variance

        sum_child_mse = 0.0

        for k in range(self.num_cluster):
            clus_data = X[clus_label == k]
            clus_gt = Y[clus_label == k]

            reg = LinearRegression().fit(clus_data, clus_gt)
            pred_Y = reg.predict(clus_data)
            mse = mean_squared_error(clus_gt, pred_Y)

            self.gtvar_child[k] = np.var(clus_gt) * len(clus_gt)  # mean as predicted Y, should use regression model
            self.source_mse_child[k] = np.mean(np.var(clus_data, axis=0))

            sum_child_mse += self.source_mse_child[k] * len(clus_data) / len(X)

        return self.source_mse_parent - sum_child_mse


class HierKmeans():
    def __init__(self, depth, min_sample_num, num_cluster, metric, standerdization=True, istop=True, topid=None,
                 topsampleidx=None, topgtvar=None, topmse=None):
        self.nodes = {}
        self.depth = depth
        self.standerdization = standerdization
        self.min_sample_num = min_sample_num
        # self.learner = learner
        self.num_cluster = num_cluster
        self.metric = metric
        # self.num_class = -1
        self.trained = False
        self.istop = istop
        self.topid = topid
        self.topsampleidx = topsampleidx
        self.topgtvar = topgtvar
        self.topmse = topmse

        self.leaf_learner = []
        self.leaf_centroid = []

    def fit(self, X, Y, feats):

        X = X.reshape(-1, X.shape[-1])

        if self.standerdization:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)

        if self.istop == True:
            tmp_data = [{'X': X, 'Y': Y, 'id': 'R', 'mse': 0}]  # leaf nodes candidates
        else:
            tmp_data = [{'X': X, 'Y': Y, 'id': self.topid, 'mse': self.topmse, 'sample_idx': self.topsampleidx,
                         'gtvar': self.topgtvar}]

        source_mse_decrease = []
        gtvar = []
        global_gtvar = []
        global_source_mse = []

        for i in range(self.depth):  # splitting的次数
            print("-----  DEPTH {} -----".format(i + 1))
            tmp = []

            if i == 0:  # root node

                tmp_node = HierNode(
                    # num_class=self.num_class,
                    num_cluster=self.num_cluster,
                    metric=self.metric,
                    isleaf=(i == self.depth - 1),
                    id=tmp_data[i]['id'])
                tmp_node.fit(tmp_data[i]['X'], tmp_data[i]['Y'])

                self.nodes['R'] = {'kmeans': tmp_node.kmeans,
                                   'leaf': 0}

                source_mse_decrease.append(
                    tmp_node.source_mse_decrease(tmp_data[i]['X'], tmp_data[i]['Y']))  # also generate gt varaince

                tmp_data[i]['mse'] = tmp_node.source_mse_parent
                tmp_data[i]['gtvar'] = tmp_node.gtvar_parent

                global_source_mse.append(tmp_node.source_mse_parent)
                # gtvar.append(tmp_node.gtvar_parent)

                label = tmp_node.predict(tmp_data[i]['X'])
                global_source_mse.append(
                    tmp_node.source_mse_child[0] * (np.sum(label == 0) / len(X)) + tmp_node.source_mse_child[1] * (
                                np.sum(label == 1) / len(X)))

                if self.istop == True:

                    for k in range(self.num_cluster):
                        idx = (label == k)
                        tmp.append(
                            {'X': tmp_data[i]['X'][idx],
                             'Y': tmp_data[i]['Y'][idx],
                             'id': tmp_data[i]['id'] + str(k),
                             'mse': tmp_node.source_mse_child[k],
                             'gtvar': tmp_node.gtvar_child[k],
                             'sample_idx': np.where(idx == True)[0]})

                        self.nodes[tmp_data[i]['id'] + str(k)] = {'X': tmp_data[i]['X'][idx],
                                                                  'Y': tmp_data[i]['Y'][idx],
                                                                  'mse': tmp_node.source_mse_child[k],
                                                                  'gtvar': tmp_node.gtvar_child[k],
                                                                  'sample_idx': np.where(idx == True)[0],
                                                                  'leaf': 1
                                                                  }
                        gtvar.append(tmp_node.gtvar_child[k])
                else:
                    print(tmp_node.source_mse_parent, self.topmse)
                    print(tmp_node.gtvar_parent, self.topgtvar)

                    for k in range(self.num_cluster):
                        idx = (label == k)
                        tmp.append(
                            {'X': tmp_data[i]['X'][idx],
                             'Y': tmp_data[i]['Y'][idx],
                             'id': tmp_data[i]['id'] + str(k),
                             'mse': tmp_node.source_mse_child[k],
                             'gtvar': tmp_node.gtvar_child[k],
                             'sample_idx': tmp_data[i]['sample_idx'][idx]})

                        self.nodes[tmp_data[i]['id'] + str(k)] = {'X': tmp_data[i]['X'][idx],
                                                                  'Y': tmp_data[i]['Y'][idx],
                                                                  'mse': tmp_node.source_mse_child[k],
                                                                  'gtvar': tmp_node.gtvar_child[k],
                                                                  'sample_idx': tmp_data[i]['sample_idx'][idx],
                                                                  'leaf': 1
                                                                  }
                        gtvar.append(tmp_node.gtvar_child[k])

                tmp_data = tmp

            if i != 0:
                # find which node to split
                tmp_mse = np.zeros((len(tmp_data)))
                for j in range(len(tmp_data)):
                    if tmp_data[j]['X'].shape[0] > self.min_sample_num:
                        tmp_mse[j] = tmp_data[j]['gtvar']
                idx_to_split = int(np.argmax(tmp_mse))

                key = tmp_data[idx_to_split]['id']

                tmp_node = HierNode(
                    # num_class=self.num_class,
                    num_cluster=self.num_cluster,
                    metric=self.metric,
                    isleaf=(i == self.depth - 1),
                    id=tmp_data[idx_to_split]['id'])

                tmp_node.fit(tmp_data[idx_to_split]['X'], tmp_data[idx_to_split]['Y'])

                self.nodes[key] = {'kmeans': tmp_node.kmeans,
                                   'leaf': 0}  # 清空要分的node里的sample, 保留kmean

                source_mse_decrease.append(
                    tmp_node.source_mse_decrease(tmp_data[idx_to_split]['X'], tmp_data[idx_to_split]['Y']))

                tmp.extend(tmp_data)
                tmp.pop(idx_to_split)

                gtvar.pop(idx_to_split)

                label = tmp_node.predict(tmp_data[idx_to_split]['X'])
                for k in range(self.num_cluster):
                    idx = (label == k)
                    print(np.sum(idx))
                    sample_idx = tmp_data[idx_to_split]['sample_idx'][idx]

                    tmp.append(
                        {'X': tmp_data[idx_to_split]['X'][idx],
                         'Y': tmp_data[idx_to_split]['Y'][idx],
                         'id': tmp_data[idx_to_split]['id'] + str(k),
                         'mse': tmp_node.source_mse_child[k],
                         'gtvar': tmp_node.gtvar_child[k],
                         'sample_idx': sample_idx})  # be careful! this idx is based on its parent node!

                    self.nodes[tmp_data[idx_to_split]['id'] + str(k)] = {'X': tmp_data[idx_to_split]['X'][idx],
                                                                         'Y': tmp_data[idx_to_split]['Y'][idx],
                                                                         'mse': tmp_node.source_mse_child[k],
                                                                         'gtvar': tmp_node.gtvar_child[k],
                                                                         'sample_idx': sample_idx,
                                                                         'leaf': 1}
                    gtvar.append(tmp_node.gtvar_child[k])

                tmp_data = tmp

                # calculate global source mse in this splitting
                global_source_mse_i = 0.0
                for i in range(len(tmp_data)):
                    global_source_mse_i += tmp_data[i]['mse'] * (len(tmp_data[i]['X']) / len(X))

                global_source_mse.append(global_source_mse_i)

            # print("decrease of MSE:", source_mse_decrease)
            # print('\n')
            print("gt variance:", gtvar)
            global_gtvar.append(np.sum(gtvar))
            print("global gt var:", global_gtvar)
            print("global source mse:", global_source_mse)

            if len(tmp) == 0 and i != self.depth - 1:
                print("       <Warning> depth %s not achieved, actual depth %s" % (str(self.depth), str(i + 1)))
                self.depth = i
                break

        for node_id in self.nodes.keys():
            if self.nodes[node_id]['leaf'] == 1:
                feat_idx = self.nodes[node_id]['sample_idx']
                leaf_feat = feats[feat_idx]
                leaf_Y = Y[feat_idx]
                reg = RandomForestRegressor(n_estimators=20, min_samples_leaf=60, min_samples_split=200).fit(leaf_feat,
                                                                                                             leaf_Y)
                self.nodes[node_id]['reg'] = reg

        # leaf_nodes_X = []
        # leaf_nodes_X_idx = []
        #
        # for node_id in self.nodes.keys():
        #     if self.nodes[node_id] != []:
        #         print("node id and leaf size:", node_id, self.nodes[node_id]['X'].shape[0])
        #
        #         leaf_nodes_X.append(self.nodes[node_id]['X'])
        #         leaf_nodes_X_idx.append(self.nodes[node_id]['sample_idx'])
        #
        #
        #
        # self.leaf_centroid = np.zeros((len(leaf_nodes_X), X.shape[-1]))
        # # leaf_learner = []
        # for k in range(len(leaf_nodes_X)):
        #     self.leaf_centroid[k] = np.mean(leaf_nodes_X[k], axis=0)
        #
        #     self.leaf_learner.append(
        #         RandomForestRegressor(n_estimators=10, min_samples_leaf=60, min_samples_split=100).fit(leaf_nodes_X[k],
        #                                                                                                leaf_nodes_Y[k]))

        self.trained = True

    def clustering(self, X):

        print('>>>>>>>>>>>>>>>>>>>>>>>Clustering')
        X = X.reshape(-1, X.shape[-1])
        if self.standerdization:
            X = self.scaler.transform(X)

        pred = np.zeros(X.shape[0])

        k = self.nodes.copy()
        key = k.keys()
        self.depth = 1
        for k in key:
            if k != 'R':
                if len(k) - 1 > self.depth:
                    self.depth = len(k) - 1
        del k
        print('>>>>>>>>>>>>>>>>>>>>>>>TREE depth = {}'.format(self.depth))

        tmp_data = []
        label = self.nodes['R']['kmeans'].predict(X)

        for k in range(2):
            tmp_data.append({'X': X[label == k], 'idx': np.arange(X.shape[0])[label == k], 'id': 'R' + str(k)})

        for i in range(self.depth):
            tmp = []
            for j in range(len(tmp_data)):
                if tmp_data[j]['X'].shape[0] == 0:
                    continue

                if self.nodes[tmp_data[j]['id']]['leaf'] == 0:
                    label = self.nodes[tmp_data[j]['id']]['kmeans'].predict(tmp_data[j]['X'])
                    for k in range(2):
                        tmp.append({'X': tmp_data[j]['X'][label == k], 'idx': tmp_data[j]['idx'][label == k],
                                    'id': tmp_data[j]['id'] + str(k)})
                else:
                    pred[tmp_data[j]['idx']] = self.nodes[tmp_data[j]['id']]['reg'].predict(
                        tmp_data[j]['X'])  # k_idx is the idx number of leaf nodes, can be regression model

            tmp_data = tmp

        return pred

    def predict_proba(self, X):
        assert (self.trained == True), "Must call fit first!"

        pred = np.zeros((X.shape[0]))
        for i in range(X.shape[0]):
            # find closest leaf centroid
            # dist = np.zeros((len(leaf_centroid)))
            # for j in range(len(leaf_centroid)):
            dist = euclidean_distances(self.leaf_centroid,
                                       X[i].reshape(1, X.shape[-1]))  # euclidean distance to centroids 11-D
            idx = int(np.argmin(dist))
            pred[i] = self.leaf_learner[idx].predict(X[i].reshape(1, X.shape[-1]))

        return pred

    def score(self, X, Y):
        Y_pred = self.predict_proba(X)

        return mean_squared_error(Y, Y_pred)


if __name__ == "__main__":
    import multiprocessing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn import preprocessing
    from matplotlib import pyplot as plt
    import pickle
    import h5py
    import sklearn.metrics as metrics
    import time

    sample_num = 154

    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/multi_resolution/features/original_resolution/'

    # h5f = h5py.File(save_dir_matFiles + 'context/train_context_1.h5', 'r')
    # context_1 = h5f['attribute'][:sample_num]
    # context_1 = context_1.reshape(-1, context_1.shape[-1])
    # print(context_1.shape)
    #
    # h5f = h5py.File(save_dir_matFiles + 'context/train_context_2.h5', 'r')
    # context_2 = h5f['attribute'][:sample_num]
    # context_2 = context_2.reshape(-1, context_2.shape[-1])
    # print(context_2.shape)
    #
    # h5f = h5py.File(save_dir_matFiles + 'context/train_context_3.h5', 'r')
    # context_3 = h5f['attribute'][:sample_num]
    # context_3 = context_3.reshape(-1, context_3.shape[-1])
    # print(context_3.shape)
    #
    # h5f = h5py.File(save_dir_matFiles + 'context/train_context_4.h5', 'r')
    # context_4 = h5f['attribute'][:sample_num]
    # context_4 = context_4.reshape(-1, context_4.shape[-1])
    # print(context_4.shape)
    #
    # h5f = h5py.File(save_dir_matFiles + 'context/train_feat_for_reg.h5', 'r')
    # train_featsforreg = h5f['attribute'][:sample_num]
    # train_featsforreg = train_featsforreg.reshape(-1, train_featsforreg.shape[-1])
    # print(train_featsforreg.shape)
    #
    #
    # h5f = h5py.File(save_dir_matFiles + 'region/train_target_modified.h5', 'r')
    # train_target = h5f['attribute'][:]
    # print(train_target.shape)
    # train_target = train_target[:sample_num].reshape(-1)
    # init_var = np.var(train_target)
    # print(train_target.shape, init_var)
    #
    # # number of positives and negatives
    # print(np.sum([train_target>0]), np.sum([train_target<0]))
    #
    #
    #
    #
    # #%% reduce samples to make train target smaller and balanced
    # pos_target = train_target[train_target > 0]
    # neg_target = train_target[train_target < 0]
    #
    # hist_p, bin_edge_p = np.histogram(pos_target, range=(0,2), bins=20)
    # hist_n, bin_edge_n = np.histogram(neg_target, range=(-2,0), bins=20)
    #
    #
    # neg_hist_max = np.zeros((len(hist_n)))
    # for k in range(len(hist_p)):
    #     neg_hist_idx = 19-k
    #     if hist_n[neg_hist_idx] > hist_p[k]:
    #         neg_hist_max[neg_hist_idx] = hist_p[k]
    #     if hist_n[neg_hist_idx] <= hist_p[k]:
    #         neg_hist_max[neg_hist_idx] = hist_n[neg_hist_idx]
    #
    #
    #
    # # select samples based on neg_hist_max limitation
    # pos_idx = np.where(train_target > 0)
    # neg_idx = []
    # count = np.zeros((18))
    # permuted = np.random.permutation(len(train_target))
    # for k in permuted:
    #
    #     if train_target[k] >= -1.8 and train_target[k] < -1.7 and count[17] < neg_hist_max[2]:
    #         count[17] = count[17] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -1.7 and train_target[k] < -1.6 and count[16] < neg_hist_max[3]:
    #         count[16] = count[16] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -1.6 and train_target[k] < -1.5 and count[15] < neg_hist_max[4]:
    #         count[15] = count[15] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -1.5 and train_target[k] < -1.4 and count[14] < neg_hist_max[5]:
    #         count[14] = count[14] + 1
    #         neg_idx.append(k)
    #
    #     if train_target[k] >= -1.4 and train_target[k] < -1.3 and count[13] < neg_hist_max[6]:
    #         count[13] = count[13] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -1.3 and train_target[k] < -1.2 and count[12] < neg_hist_max[7]:
    #         count[12] = count[12] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -1.2 and train_target[k] < -1.1 and count[11] < neg_hist_max[8]:
    #         count[11] = count[11] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -1.1 and train_target[k] < -1.0 and count[10] < neg_hist_max[9]:
    #         count[10] = count[10] + 1
    #         neg_idx.append(k)
    #
    #     if train_target[k] >= -1.0 and train_target[k] < -0.9 and count[9] < neg_hist_max[10]:
    #         count[9] = count[9] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -0.9 and train_target[k] < -0.8 and count[8] < neg_hist_max[11]:
    #         count[8] = count[8] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -0.8 and train_target[k] < -0.7 and count[7] < neg_hist_max[12]:
    #         count[7] = count[7] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -0.7 and train_target[k] < -0.6 and count[6] < neg_hist_max[13]:
    #         count[6] = count[6] + 1
    #         neg_idx.append(k)
    #
    #     if train_target[k] >= -0.6 and train_target[k] < -0.5 and count[5] < neg_hist_max[14]:
    #         count[5] = count[5] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -0.5 and train_target[k] < -0.4 and count[4] < neg_hist_max[15]:
    #         count[4] = count[4] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -0.4 and train_target[k] < -0.3 and count[3] < neg_hist_max[16]:
    #         count[3] = count[3] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -0.3 and train_target[k] < -0.2 and count[2] < neg_hist_max[17]:
    #         count[2] = count[2] + 1
    #         neg_idx.append(k)
    #
    #     if train_target[k] >= -0.2 and train_target[k] < -0.1 and count[1] < neg_hist_max[18]:
    #         count[1] = count[1] + 1
    #         neg_idx.append(k)
    #     if train_target[k] >= -0.1 and train_target[k] < 0 and count[0] < neg_hist_max[19]:
    #         count[0] = count[0] + 1
    #         neg_idx.append(k)
    #
    #
    # neg_target_reduced = train_target[neg_idx]
    # print(len(neg_target_reduced), len(pos_target))

    # plt.figure(0)
    # bin_edge = np.linspace(0, 2, 21, endpoint= True)
    # n_p, bins_p, patches_p = plt.hist(x=pos_target, bins=bin_edge, color='b',
    #                                   alpha=0.7, rwidth=0.85)
    # bin_edge_2 = np.linspace(-2, 0, 21, endpoint=True)
    # n_n, bins_n, patches_n = plt.hist(x=neg_target_reduced, bins=bin_edge_2, color='y',
    #                                   alpha=0.7, rwidth=0.85)
    #
    # plt.grid(axis='y', alpha=0.75)
    # plt.title('histogram of all images modified GC-reduced negative ')
    #
    # plt.savefig(save_dir_matFiles + "region/leaf_target_hist/train_target_modified_no-1.png")
    # plt.close(0)

    # pos_context_1 = context_1[train_target>0]
    # pos_context_2 = context_2[train_target>0]
    # pos_context_3 = context_3[train_target>0]
    # pos_context_4 = context_4[train_target>0]
    # pos_featforreg = train_featsforreg[train_target >0]
    #
    # neg_context_1 = context_1[neg_idx]
    # neg_context_2 = context_2[neg_idx]
    # neg_context_3 = context_3[neg_idx]
    # neg_context_4 = context_4[neg_idx]
    # neg_featforreg = train_featsforreg[neg_idx]
    #
    # pickle.dump(pos_target, open(save_dir_matFiles + 'region/pos_target_region.pkl', 'wb'))
    # pickle.dump(neg_target_reduced, open(save_dir_matFiles + 'region/neg_target_region_reduced.pkl', 'wb'))
    #
    # pickle.dump(pos_context_1, open(save_dir_matFiles+'region/pos_context_1_region.pkl', 'wb'))
    # pickle.dump(pos_context_2, open(save_dir_matFiles+'region/pos_context_2_region.pkl', 'wb'))
    # pickle.dump(pos_context_3, open(save_dir_matFiles+'region/pos_context_3_region.pkl', 'wb'))
    # pickle.dump(pos_context_4, open(save_dir_matFiles+'region/pos_context_4_region.pkl', 'wb'))
    #
    # pickle.dump(neg_context_1, open(save_dir_matFiles + 'region/neg_context_1_region.pkl', 'wb'))
    # pickle.dump(neg_context_2, open(save_dir_matFiles + 'region/neg_context_2_region.pkl', 'wb'))
    # pickle.dump(neg_context_3, open(save_dir_matFiles + 'region/neg_context_3_region.pkl', 'wb'))
    # pickle.dump(neg_context_4, open(save_dir_matFiles + 'region/neg_context_4_region.pkl', 'wb'))

    # pickle.dump(pos_featforreg, open(save_dir_matFiles+'region/pos_featforreg_region.pkl', 'wb'))
    # pickle.dump(neg_featforreg, open(save_dir_matFiles+'region/neg_featforreg_region.pkl', 'wb'))

    # %%
    # debug use
    # idx_n = np.random.permutation(2573842)
    # idx_p = np.random.permutation(2610641)
    # pickle.dump(idx_p, open(save_dir_matFiles + 'region/pos_idx_2610641.pkl', 'wb'))
    # pickle.dump(idx_n, open(save_dir_matFiles + 'region/neg_idx_2573842.pkl', 'wb'))

    pos_context_1 = np.load('pos_context_1.npy')
    pos_context_2 = np.load('pos_context_2.npy')
    pos_context_3 = np.load('pos_context_3.npy')

    neg_context_1 = np.load('neg_context_1.npy')
    neg_context_2 = np.load('neg_context_2.npy')
    neg_context_3 = np.load('neg_context_3.npy')

    # hop 1 for reg
    pos_featforreg = np.load('pos_context_1.npy')
    neg_featforreg = np.load('neg_context_1.npy')

    pos_target = np.ones((pos_context_1.shape[0]))
    neg_target = np.zeros((neg_context_1.shape[0]))

    context_1 = np.concatenate((pos_context_1, neg_context_1), axis=0)
    context_2 = np.concatenate((pos_context_2, neg_context_2), axis=0)
    context_3 = np.concatenate((pos_context_3, neg_context_3), axis=0)

    feats = np.concatenate((pos_featforreg, neg_featforreg), axis=0)
    train_target = np.concatenate((pos_target, neg_target), axis=0)

    init_var = np.var(train_target)
    print("initial variance:", init_var)

    # use concatenation of context_1 and context_4 to construct top of the tree
    context_top = np.concatenate((context_1, context_2, context_3), axis=1)

    print("top tree input feature shape: %s" % str(context_top.shape))

    # # PCA on context
    # pca = PCA(n_components=0.99999, svd_solver='full').fit(context_top)
    # context_scaled = pca.transform(context_top)

    t1 = time.time()
    metric = {'min_num_sample': 15000,
              'purity': 0.9}

    clf_top = HierKmeans(depth=10, min_sample_num=2000, num_cluster=2, metric=metric, istop=True)
    clf_top.fit(context_top, train_target, feats)
    print(clf_top.nodes.keys())

    print("Time for Hierarchical Kmean:", time.time() - t1)

    leaf_nodes_id = []
    leaf_nodes_mse = []
    leaf_nodes_gtvar = []
    leaf_nodes_X_idx = []

    for node_id in clf_top.nodes.keys():
        if clf_top.nodes[node_id] != []:
            print("node id and leaf size:", node_id, clf_top.nodes[node_id]['X'].shape[0])

            leaf_nodes_id.append(node_id)
            leaf_nodes_mse.append(clf_top.nodes[node_id]['mse'])
            leaf_nodes_gtvar.append(clf_top.nodes[node_id]['gtvar'])
            leaf_nodes_X_idx.append(clf_top.nodes[node_id]['sample_idx'])

    # check sample_idx for all leaf nodes, make sure no overlap:
    leaf_idx_set = [[] for i in range(len(leaf_nodes_X_idx))]
    whole_leaf_idx_check = set()

    for i in range(len(leaf_nodes_X_idx)):
        if len(set(leaf_nodes_X_idx[i])) == len(leaf_nodes_X_idx[i]):
            set_i = set(leaf_nodes_X_idx[i])

            whole_leaf_idx_check = whole_leaf_idx_check.union(set_i)

    if len(whole_leaf_idx_check) == context_top.shape[0]:
        print("!!!! Top of tree constructed !!!!")

    del leaf_idx_set, whole_leaf_idx_check
    # del clf_top

    # check variance in top tree leaf nodes
    top_num_samples_small_var = 0
    top_leaf_variance = []
    top_leaf_target_ = [[] for i in range(len(leaf_nodes_X_idx))]
    top_leaf_context_ = [[] for i in range(len(leaf_nodes_X_idx))]
    top_leaf_centroid_ = np.zeros((len(leaf_nodes_X_idx), context_top.shape[-1]))

    for k in range(len(leaf_nodes_X_idx)):
        top_leaf_idx = leaf_nodes_X_idx[k]

        top_leaf_context = context_top[top_leaf_idx]
        top_leaf_context_[k] = top_leaf_context
        top_leaf_centroid_[k] = np.mean(top_leaf_context, axis=0)

        top_leaf_target = train_target[top_leaf_idx]
        top_leaf_target_[k] = top_leaf_target

        top_leaf_variance.append(np.var(top_leaf_target))
        if np.var(top_leaf_target) < init_var:
            top_num_samples_small_var += len(top_leaf_target)

    # save leaf node information
    pickle.dump(top_leaf_context_, open(save_dir_matFiles + 'region/loss_leaf_context_.pkl', 'wb'))
    pickle.dump(top_leaf_target_, open(save_dir_matFiles + 'region/loss_leaf_target_.pkl', 'wb'))
    pickle.dump(top_leaf_centroid_, open(save_dir_matFiles + 'region/loss_leaf_centroid_.pkl', 'wb'))

    top_leaf_variance = np.array(top_leaf_variance)
    #
    # print(np.sum(top_leaf_variance > init_var), np.sum(top_leaf_variance < init_var))
    # print(top_num_samples_small_var)
    #
    #
    # plt.figure(0)
    # bin_edge = np.linspace(0,1, 81, endpoint=True)
    # n_p, bins_p, patches_p = plt.hist(x=top_leaf_variance, bins=bin_edge, color='b',
    #                                   alpha=0.7, rwidth=0.85)
    # # n_n, bins_n, patches_n = plt.hist(x=top_leaf_variance[top_leaf_variance < init_var], bins='auto', color='y',
    # #                                   alpha=0.7, rwidth=0.85)
    #
    # plt.grid(axis='y', alpha=0.75)
    # plt.title('histogram of leaf node GT variance')
    #
    # plt.savefig(save_dir_matFiles + "region/leaf_target_hist/HierKM_topleaf_GTvar.png")
    # plt.close(0)
    #
    sorted_var_idx = np.argsort(top_leaf_variance)
    sorted_var = top_leaf_variance[sorted_var_idx]
    print("sorted variance:", sorted_var)

    # # plot GT histogram in each leaf node according to variance
    #
    # for i in range(len(top_leaf_variance)):
    #     samp = top_leaf_target_[sorted_var_idx[i]]
    #
    #     # weak = []
    #     # for k in range(len(samp)):
    #     #     if samp[k] > -0.2 and samp[k] < 0.2:
    #     #         weak.append(samp[k])
    #
    #     plt.figure(0)
    #     bin_edge = np.linspace(-1, 1, 41, endpoint=True)
    #     n_p, bins_p, patches_p = plt.hist(x=samp[samp > 0], bins=bin_edge, color='b',
    #                                       alpha=0.7, rwidth=0.85, label="spliced pixels")
    #     n_n, bins_n, patches_n = plt.hist(x=samp[samp < 0], bins=bin_edge, color='y',
    #                                       alpha=0.7, rwidth=0.85, label="authentic pixels")
    #     # n_w, bins_w, patches_w = plt.hist(x=weak, bins=bin_edge, color='r',
    #     #                                   alpha=0.7, rwidth=0.85, label="background pixels")
    #
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.title('histogram in leaf node {}, samp_num:{}, variance={:0.4f}, mean={:0.4f}'.format(i, len(samp), sorted_var[i], np.mean(samp)))
    #     plt.legend()
    #
    #     plt.savefig(save_dir_matFiles + "region/leaf_target_hist/leaf_hist_{}.png".format(i))
    #     plt.close(0)

    # %% bottom of tree

    #
    # context_bottom = np.concatenate((context_2, context_3), axis=3)
    # context_bottom = context_bottom.reshape(-1, context_bottom.shape[-1])
    #
    # bottom_leaf_idx = []
    # for k in range(len(leaf_nodes_X_idx)):
    #     start_node_id = leaf_nodes_id[k]
    #     start_node_mse = leaf_nodes_mse[k]
    #     start_node_gtvar = leaf_nodes_gtvar[k]
    #
    #     leaf_samp_idx = leaf_nodes_X_idx[k]
    #     leaf_samp_context123 = context_bottom[leaf_samp_idx]
    #     leaf_samp_gt = train_target[leaf_samp_idx]
    #
    #     print("--- start training sub-tree for leaf node {} ---".format(start_node_id))
    #
    #     metric_bot = {'min_num_sample': 500,'purity': 0.9}
    #     clf_bot = HierKmeans(depth=15, min_sample_num=3000, num_cluster=2, metric=metric_bot, istop = False, topid=start_node_id, topmse=start_node_mse, topgtvar=start_node_gtvar, topsampleidx=leaf_samp_idx)
    #
    #     clf_bot.fit(leaf_samp_context123, leaf_samp_gt)
    #
    #     print("--- finish training sub-tree for leaf node {} ---".format(start_node_id))
    #
    #     for node_id in clf_bot.nodes.keys():
    #         if clf_bot.nodes[node_id] != []:
    #             print("--- node id and leaf size:", node_id, clf_bot.nodes[node_id]['X'].shape[0])
    #
    #             bottom_leaf_idx.append(clf_bot.nodes[node_id]['sample_idx'])
    #
    # print("!!!! Bottom of tree constructed !!!!")
    #
    #
    #
    #
    #
    # variance = np.zeros((len(bottom_leaf_idx)))
    # num_samples_small_var = 0
    # leaf_target = [[] for i in range(len(bottom_leaf_idx))]
    # for i in range(len(bottom_leaf_idx)):
    #     leaf_idx = bottom_leaf_idx[i]
    #     leaf_target_tmp = train_target[leaf_idx]
    #     leaf_target[i] = leaf_target_tmp
    #     variance[i] = np.var(leaf_target_tmp)
    #     if np.var(leaf_target_tmp) < init_var:
    #         num_samples_small_var += len(leaf_target_tmp)
    #
    #
    # print(np.sum(variance > init_var), np.sum(variance < init_var))
    # print(num_samples_small_var)
    #
    # sorted_var_idx = np.argsort(variance)
    # sorted_var = variance[sorted_var_idx]
    #
    #
    #
    # # plot the histogram of variance in leaf nodes
    #
    # variance = np.array(variance)
    # print(np.sum(variance > init_var), np.sum(variance < init_var))
    # bin_edge = np.linspace(0, 1, 81, endpoint=True)
    # plt.figure(0)
    # n_p, bins_p, patches_p = plt.hist(x=variance, bins='auto', color='b',
    #                                   alpha=0.7, rwidth=0.85)
    #
    #
    # plt.grid(axis='y', alpha=0.75)
    # plt.title('histogram of leaf node GT variance')
    #
    # plt.savefig(save_dir_matFiles + "original_resolution/context/leaf_target_hist/HierKM_leafnodes_GTvar.png")
    # plt.close(0)
    #
    #
    #
    # for i in range(len(variance)):
    #     samp = leaf_target[sorted_var_idx[i]]
    #
    #     weak = []
    #     for k in range(len(samp)):
    #         if samp[k] > -0.2 and samp[k] < 0.2:
    #             weak.append(samp[k])
    #
    #     plt.figure(0)
    #     bin_edge = np.linspace(-1, 1, 41, endpoint=True)
    #     n_p, bins_p, patches_p = plt.hist(x=samp[samp > 0.2], bins=bin_edge, color='b',
    #                                       alpha=0.7, rwidth=0.85, label="spliced pixels")
    #     n_n, bins_n, patches_n = plt.hist(x=samp[samp < -0.2], bins=bin_edge, color='y',
    #                                       alpha=0.7, rwidth=0.85, label="authentic pixels")
    #     n_w, bins_w, patches_w = plt.hist(x=weak, bins=bin_edge, color='r',
    #                                       alpha=0.7, rwidth=0.85, label="background pixels")
    #
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.title('histogram of train target in leaf node {}, variance = {}'.format(i, sorted_var[i]))
    #     plt.legend()
    #
    #     plt.savefig(save_dir_matFiles + "original_resolution/context/leaf_target_hist/leaf_hist_{}.png".format(i))
    #     plt.close(0)
    #
    #
    # print("------- DONE -------\n")

    # %% train random forest in each leaf node

    top_leaf_context_ = pickle.load(open(save_dir_matFiles + 'region/loss_leaf_context_.pkl', 'rb'))
    top_leaf_target_ = pickle.load(open(save_dir_matFiles + 'region/loss_leaf_target_.pkl', 'rb'))
    top_leaf_centroid_ = pickle.load(open(save_dir_matFiles + 'region/loss_leaf_centroid_.pkl', 'rb'))

    regs = []
    train_pred_target = []
    train_target = []
    for k in range(len(top_leaf_target_)):
        tmp_target = top_leaf_target_[k]
        tmp_context = top_leaf_context_[k]

        if len(tmp_target) > 202:
            reg = RandomForestRegressor(n_estimators=20, min_samples_leaf=100, min_samples_split=200).fit(tmp_context,
                                                                                                          tmp_target)
            regs.append(reg)

            tmp_pred_target = reg.predict(tmp_context)
            # if len(tmp_target) <= 2000 and len(tmp_target) >= 202:
            #     reg = LinearRegression().fit(tmp_context, tmp_target)
            #     regs.append(reg)

            tmp_pred_target = reg.predict(tmp_context)

        if len(tmp_context) <= 202:
            reg = np.mean(tmp_target)
            regs.append(reg)

            tmp_pred_target = []
            for i in range(len(tmp_target)):
                tmp_pred_target.append(reg)
            tmp_pred_target = np.array(tmp_pred_target)

        train_pred_target.extend(tmp_pred_target)
        train_target.extend(tmp_target)

    train_pred_target = np.array(train_pred_target)
    train_target = np.array(train_target)

    # per class accuracy
    train_label = np.zeros((train_target.shape[0]))
    train_label[train_target > 0] = 1

    train_pred_label = np.zeros((train_target.shape[0]))
    train_pred_label[train_pred_target > 0] = 1

    plt.figure()
    n_p, bins_p, patches_p = plt.hist(x=train_pred_target[train_target > 0], bins='auto', color='b', rwidth=0.9,
                                      label="spliced region")
    n_n, bins_n, patches_n = plt.hist(x=train_pred_target[train_target < 0], bins='auto', color='y', rwidth=0.9,
                                      label="authentic region")

    plt.grid(axis='y')
    plt.xlabel('Predicted Y')
    plt.ylabel('Frequency')
    plt.title('histogram of Training predicted Y - region')
    plt.legend()

    plt.savefig(save_dir_matFiles + "region/pred_result/hist_1-1_lab_HKM_train_region.png")

    C_train = metrics.confusion_matrix(train_label, train_pred_label, labels=[0, 1])
    per_class_accuracy_train = np.diag(C_train.astype(np.float32)) / np.sum(C_train.astype(np.float32), axis=1)
    print("train:", per_class_accuracy_train)

    pickle.dump(regs, open(save_dir_matFiles + 'region/loss_reg_model_30node.pkl', 'wb'))

    # %%
    print(" ------- Test Process -------")

    regs = pickle.load(open(save_dir_matFiles + 'region/loss_reg_model_30node.pkl', 'rb'))
    top_leaf_context_ = pickle.load(open(save_dir_matFiles + 'region/loss_leaf_context_.pkl', 'rb'))
    top_leaf_target_ = pickle.load(open(save_dir_matFiles + 'region/loss_leaf_target_.pkl', 'rb'))
    top_leaf_centroid_ = pickle.load(open(save_dir_matFiles + 'region/loss_leaf_centroid_.pkl', 'rb'))

    # read text images context vector
    test_sample_num = 1

    with open(
            '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/' + 'filenames_columbia104.pkl',
            'rb') as fid:
        test_filenames = pickle.load(fid)

    test_filenames = test_filenames[:test_sample_num]

    h5f = h5py.File(save_dir_matFiles + 'context/test_context_1.h5', 'r')
    test_context_1 = h5f['attribute'][:test_sample_num]
    test_context_1 = test_context_1.reshape(-1, test_context_1.shape[-1])
    print(test_context_1.shape)

    h5f = h5py.File(save_dir_matFiles + 'context/test_context_2.h5', 'r')
    test_context_2 = h5f['attribute'][:test_sample_num]
    test_context_2 = test_context_2.reshape(-1, test_context_2.shape[-1])
    print(test_context_2.shape)

    h5f = h5py.File(save_dir_matFiles + 'context/test_context_3.h5', 'r')
    test_context_3 = h5f['attribute'][:test_sample_num]
    test_context_3 = test_context_3.reshape(-1, test_context_3.shape[-1])
    print(test_context_3.shape)

    h5f = h5py.File(save_dir_matFiles + 'context/test_context_4.h5', 'r')
    test_context_4 = h5f['attribute'][:test_sample_num]
    test_context_4 = test_context_4.reshape(-1, test_context_4.shape[-1])
    print(test_context_4.shape)

    h5f = h5py.File(save_dir_matFiles + 'context/test_target.h5', 'r')
    test_target = h5f['attribute'][:test_sample_num]
    test_target = test_target.reshape(-1)
    print(test_target.shape)

    h5f = h5py.File(save_dir_matFiles + 'context/test_feat_for_reg.h5', 'r')
    test_featsforreg = h5f['attribute'][:test_sample_num]
    test_featsforreg = test_featsforreg.reshape(-1, test_featsforreg.shape[-1])
    print(test_featsforreg.shape)

    p_idx = np.where(test_target == 1)[0][:1000].tolist()
    n_idx = np.where(test_target == 0)[0][:1000].tolist()
    p_idx.extend(n_idx)

    test_context_1 = test_context_1[p_idx]
    test_context_2 = test_context_2[p_idx]
    test_context_3 = test_context_3[p_idx]
    test_context_4 = test_context_4[p_idx]
    test_target = test_target[p_idx]

    test_context = np.concatenate((test_context_1, test_context_2, test_context_3, test_context_4), axis=1)

    start = time.time()
    # test_pred_target = []
    # for k in range(test_context.shape[0]):
    #     dist = euclidean_distances(top_leaf_centroid_, test_scaled[k].reshape(1, test_scaled.shape[-1]))  # euclidean distance to centroids 11-D
    #
    #     idx = int(np.argmin(dist))
    #
    #     if type(regs[idx]) is np.float64:
    #         pred = regs[idx]
    #
    #         pred = np.array([pred])
    #         print("！！！", k, pred, pred.dtype, type(pred))
    #
    #     else:
    #         pred = regs[idx].predict(test_scaled[k].reshape(1, test_scaled.shape[-1]))
    #         # print(pred, pred.dtype, type(pred))
    #
    #     test_pred_target.extend(pred)

    num_cores = int(multiprocessing.cpu_count() / 2)

    test_pred_target = Parallel(n_jobs=num_cores, backend='multiprocessing')(
        delayed(test_predict)(test_context, top_leaf_centroid_, regs, k) for k in range(test_context.shape[0]))

    test_pred_target = np.array(test_pred_target)[:, 0]

    end1 = time.time()
    print("Time for predict:", end1 - start)

    pos_pred = test_pred_target[test_target > 0]
    neg_pred = test_pred_target[test_target == 0.0]

    # per class accuracy
    test_label = np.zeros((test_target.shape[0]))
    test_label[test_target > 0] = 1

    test_pred_label = np.zeros((test_target.shape[0]))
    test_pred_label[test_pred_target > 0] = 1

    C_test = metrics.confusion_matrix(test_label, test_pred_label, labels=[0, 1])
    per_class_accuracy_test = np.diag(C_test.astype(np.float32)) / np.sum(C_test.astype(np.float32), axis=1)
    print("test:", per_class_accuracy_test)

    end2 = time.time()
    print("Time for per class acc:", end2 - end1)

    plt.figure(0)
    n_p, bins_p, patches_p = plt.hist(x=pos_pred, bins='auto', color='b', rwidth=0.9,
                                      label="spliced region")
    n_n, bins_n, patches_n = plt.hist(x=neg_pred, bins='auto', color='y', rwidth=0.9,
                                      label="authentic region")

    plt.grid(axis='y')
    plt.xlabel('Predicted Y')
    plt.ylabel('Frequency')
    plt.title('histogram of Test predicted Y - region')
    plt.legend()

    plt.savefig(save_dir_matFiles + "region/pred_result/histogram_1-1_lab_HierKM_test_region.png")
    plt.close(0)

    end3 = time.time()
    print("Time for drawing:", end3 - end2)

    test_outprobmap = np.reshape(test_pred_target, (test_sample_num, 256, 384))

    # test_outprobmap = pickle.load(open(save_dir_matFiles + 'region/test_output_prob_map_forfuse.pkl', 'rb'))

    for k in range(test_sample_num):
        opmap = test_outprobmap[k]

        plt.figure(0)
        plt.imshow(opmap, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.axis('off')
        plt.savefig(
            save_dir_matFiles + 'region/pred_result/' + test_filenames[k][:-4] + '_opmap_1-1_lab_HKM_region_loss.png')

        plt.close(0)

    pickle.dump(test_outprobmap, open(save_dir_matFiles + 'region/test_output_prob_map_loss.pkl', 'wb'))

    plt.figure(0)
    n_p, bins_p, patches_p = plt.hist(x=pos_pred, bins='auto', color='b', rwidth=0.9,
                                      label="spliced region")
    n_n, bins_n, patches_n = plt.hist(x=neg_pred, bins='auto', color='y', rwidth=0.9,
                                      label="authentic region")

    plt.grid(axis='y')
    plt.xlabel('Predicted Y')
    plt.ylabel('Frequency')
    plt.title('histogram of Test predicted Y - region')
    plt.legend()

    plt.savefig(save_dir_matFiles + "region/pred_result/histogram_1-1_lab_HierKM_test_region.png")
    plt.close(0)

    end3 = time.time()
    print("Time for drawing:", end3 - end2)
