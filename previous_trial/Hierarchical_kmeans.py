# 2020.04.23
# Hierachical Kmeans
# author: yifan
# modified by Yao Zhu


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error


# from mylearner import myLearner

class HierNode():
    def __init__(self, learner, num_cluster, metric, isleaf=False, id='R'):
        self.learner = learner
        self.kmeans = KMeans(n_clusters=num_cluster)
        self.num_cluster = num_cluster
        self.metric = metric
        self.isleaf = isleaf
        self.id = id
        self.centroid = []
        self.mse_parent = 0.0
        self.mse_child = np.zeros((num_cluster))

    def metric_(self, X, Y):
        # if 'func' in self.metric.keys():
        #     return self.metric['func'](X, self.metric)
        if X.shape[0] < self.num_cluster * self.metric['min_num_sample']:
            return True
        if len(Y[Y>0])/len(Y) > self.metric['purity'] or len(Y[Y<0])/len(Y) > self.metric['purity']:
            return True

        return False

    def fit(self, X, Y):
        self.kmeans.fit(X)
        # if self.metric_(X, Y) == True:
        #     self.isleaf = True
        # if self.isleaf == True:
        #     self.learner.fit(X, Y)
    
    def predict(self, X):
        # if self.isleaf == True:
        #     try:
        #         prob = self.learner.predict_proba(X)
        #     except:
        #         prob = self.learner.predict(X)
        #     return prob
        # else:
        # self.centroid = self.kmeans.cluster_centers_
        return self.kmeans.predict(X)


    def mse_decrease(self, X,Y):
        clus_label = self.predict(X)
        reg = self.learner.fit(X,Y)

        # self.mse_parent = self.learner.mse(X,Y)
        self.mse_parent = mean_squared_error(Y, reg.predict(X))
        sum_child_mse = 0.0

        for k in range(self.num_cluster):
            clus_data = X[clus_label == k]
            clus_target = Y[clus_label == k]
            reg = self.learner.fit(clus_data, clus_target)
            # self.mse_child[k] = self.learner.mse(clus_data, clus_target)
            self.mse_child[k] = mean_squared_error(clus_target, reg.predict(clus_data))
            sum_child_mse += self.mse_child[k] * len(clus_target)/len(Y)

        return self.mse_parent - sum_child_mse




class HierKmeans():
    def __init__(self, depth, learner, num_cluster, metric):
        self.nodes = {}
        self.depth = depth
        self.learner = learner
        self.num_cluster = num_cluster
        self.metric = metric
        # self.num_class = -1
        self.trained = False
        self.leaf_learner = []
        self.leaf_centroid = []

    def fit(self, X, Y):
        # self.num_class = 2
        tmp_data = [{'X':X, 'Y':Y, 'id':'R', 'mse':0}]    # leaf nodes candidates
        mse_decrease = []


        for i in range(self.depth):  # splitting的次数


            tmp = []

            if i == 0:   # root node
                self.nodes['R'] = []

                tmp_node = HierNode(learner=self.learner,
                                    # num_class=self.num_class,
                                    num_cluster=self.num_cluster,
                                    metric=self.metric,
                                    isleaf=(i == self.depth - 1),
                                    id=tmp_data[i]['id'])
                tmp_node.fit(tmp_data[i]['X'], tmp_data[i]['Y'])

                mse_decrease.append(tmp_node.mse_decrease(tmp_data[i]['X'], tmp_data[i]['Y']))

                tmp_data[i]['mse'] = tmp_node.mse_parent

                label = tmp_node.predict(tmp_data[i]['X'])
                for k in range(self.num_cluster):
                    idx = (label == k)
                    tmp.append({'X':tmp_data[i]['X'][idx], 'Y':tmp_data[i]['Y'][idx], 'id':tmp_data[i]['id']+str(k), 'mse': tmp_node.mse_child[k]})
                    self.nodes[tmp_data[i]['id']+str(k)] = {'X':tmp_data[i]['X'][idx], 'Y':tmp_data[i]['Y'][idx]}


                tmp_data = tmp

            if i != 0:
                # find which node to split
                tmp_mse = np.zeros((len(tmp_data)))
                for j in range(len(tmp_data)):
                    if tmp_data[j]['Y'].shape[0] > 20000:
                        tmp_mse[j] = tmp_data[j]['mse']
                idx_to_split = int(np.argmax(tmp_mse))

                key = tmp_data[idx_to_split]['id']
                self.nodes[key] = []

                tmp_node = HierNode(learner=self.learner,
                                    # num_class=self.num_class,
                                    num_cluster=self.num_cluster,
                                    metric=self.metric,
                                    isleaf=(i == self.depth - 1),
                                    id=tmp_data[idx_to_split]['id'])

                tmp_node.fit(tmp_data[idx_to_split]['X'], tmp_data[idx_to_split]['Y'])
                mse_decrease.append(tmp_node.mse_decrease(tmp_data[idx_to_split]['X'], tmp_data[idx_to_split]['Y']))

                tmp.extend(tmp_data)
                tmp.pop(idx_to_split)

                label = tmp_node.predict(tmp_data[idx_to_split]['X'])
                for k in range(self.num_cluster):
                    idx = (label == k)
                    tmp.append(
                        {'X': tmp_data[idx_to_split]['X'][idx], 'Y': tmp_data[idx_to_split]['Y'][idx], 'id': tmp_data[idx_to_split]['id'] + str(k),
                         'mse': tmp_node.mse_child[k]})
                    self.nodes[tmp_data[idx_to_split]['id'] + str(k)] = {'X': tmp_data[idx_to_split]['X'][idx], 'Y': tmp_data[idx_to_split]['Y'][idx], 'mse': tmp_node.mse_child[k]}

                tmp_data = tmp

            print("decrease of MSE:", mse_decrease)


            # for j in range(len(tmp_data)):     # 每一个child node 都会下分， 不对
            #     if tmp_data[j]['X'].shape[0] == 0:
            #         continue
            #
            #     tmp_node = HierNode(learner=self.learner,
            #                         num_cluster=self.num_cluster,
            #                         metric=self.metric,
            #                         isleaf=(i==self.depth-1),
            #                         id=tmp_data[j]['id'])
            #     tmp_node.fit(tmp_data[j]['X'], tmp_data[j]['Y'])
            #     label = tmp_node.predict(tmp_data[j]['X'])
            #     self.nodes[tmp_data[j]['id']] = copy.deepcopy(tmp_node)
            #     if tmp_node.isleaf == True:
            #         continue
            #     for k in range(self.num_cluster):
            #         idx = (label == k)
            #         tmp.append({'X':tmp_data[j]['X'][idx], 'Y':tmp_data[j]['Y'][idx], 'id':tmp_data[j]['id']+str(k)})


            if len(tmp) == 0 and i != self.depth-1:
                print("       <Warning> depth %s not achieved, actual depth %s"%(str(self.depth), str(i+1)))
                self.depth = i
                break

        self.trained = True

        leaf_nodes_X = []
        leaf_nodes_Y = []
        for node_id in self.nodes.keys():
            if self.nodes[node_id] != []:
                tmp_y = self.nodes[node_id]['Y']

                print("node id and leaf size:", node_id, self.nodes[node_id]['Y'].shape[0])
                print("                      ", len(tmp_y[tmp_y>0])/len(tmp_y))
                leaf_nodes_X.append(self.nodes[node_id]['X'])
                leaf_nodes_Y.append(self.nodes[node_id]['Y'])

        # sample weight for each leaf node
        bin_edge = np.linspace(-1, 1, 21, endpoint=True)
        sample_weight = [[] for k in range(len(leaf_nodes_X))]

        for k in range(len(leaf_nodes_X)):
            counts, _ = np.histogram(leaf_nodes_Y[k], bins=bin_edge)
            freq = counts / leaf_nodes_Y[k].shape[0]
            sample_weight[k] = np.zeros((leaf_nodes_Y[k].shape[0]))


            idx = np.zeros((len(counts) + 1), dtype=int)
            for i in range(len(counts) + 1):
                idx[i] = np.sum(counts[:i])

            for i in range(len(idx) - 1):
                for j in range(idx[i], idx[i + 1]):
                    sample_weight[k][j] = 1 / max(freq[i], 0.01)



        self.leaf_centroid = np.zeros((len(leaf_nodes_X), X.shape[-1]))
        # leaf_learner = []
        for k in range(len(leaf_nodes_X)):
            self.leaf_centroid[k] = np.mean(leaf_nodes_X[k], axis=0)

            # self.leaf_learner.append(self.learner.fit(leaf_nodes_X[k], leaf_nodes_Y[k]))
            self.leaf_learner.append(RandomForestRegressor(n_estimators=10, min_samples_leaf=60, min_samples_split=100).fit(leaf_nodes_X[k], leaf_nodes_Y[k]))



    
    def predict_proba(self, X):
        assert (self.trained == True), "Must call fit first!"


        # pred = Parallel(n_jobs=8, backend='multiprocessing')(
        #     delayed(self.predict_proba_parallel)(X, k) for k in range(X.shape[0]))
        # pred = np.array(pred)

        pred = np.zeros((X.shape[0]))
        for i in range(X.shape[0]):
            # find closest leaf centroid
            # dist = np.zeros((len(leaf_centroid)))
            # for j in range(len(leaf_centroid)):
            dist = euclidean_distances(self.leaf_centroid, X[i].reshape(1, X.shape[-1]))   # euclidean distance to centroids 11-D
            idx = int(np.argmin(dist))
            pred[i] = self.leaf_learner[idx].predict(X[i].reshape(1, X.shape[-1]))

        return pred


    # def predict_proba_parallel(self, X, k):
    #     samp = X[k].reshape(1, X.shape[-1])
    #     dist = euclidean_distances(self.leaf_centroid, samp)
    #     idx = int(np.argmin(dist))
    #     pred_k = self.leaf_learner[idx].predict(samp)
    #
    #     return pred_k




    def score(self, X, Y):
        Y_pred = self.predict_proba(X)

        return mean_squared_error(Y, Y_pred)


if __name__ == "__main__":
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split
    import pickle

    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/multi_resolution/features/'
    positive = np.load(save_dir_matFiles + '1-2_resolution/ssl/pos_1_100.npy')[:]
    negative = pickle.load(open(save_dir_matFiles + '1-2_resolution/ssl/neg_1_100.pkl', 'rb'))[:]
    pos_target = np.load(save_dir_matFiles + '1-2_resolution/ssl/pos_target.npy')[:]
    neg_target = pickle.load(open(save_dir_matFiles + '1-2_resolution/ssl/neg_target.pkl', 'rb'))[:]
    X = np.concatenate((positive, negative), axis=0)
    Y_target = np.concatenate((pos_target, neg_target), axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_target, test_size=0.1, random_state=42)


    print(" > This is a test example: ")

    # pos_train, pos_test, y_pos_train, y_pos_test = train_test_split(positive, pos_target, test_size=0.1, random_state=42)
    # neg_train, neg_test, y_neg_train, y_neg_test = train_test_split(negative, neg_target, test_size=0.1, random_state=42)
    #
    # X_train = np.concatenate((pos_train, neg_train), axis=0)
    # X_test = np.concatenate((pos_test, neg_test), axis=0)
    #
    # Y_train = np.concatenate((y_pos_train, y_neg_train), axis=0)
    # Y_test = np.concatenate((y_pos_test, y_neg_test), axis=0)

    print(" input feature shape: %s"%str(X_train.shape))

    
    metric = {'min_num_sample':100,
              'purity':0.9}
    clf = HierKmeans(depth=45, learner=LinearRegression(), num_cluster=2, metric=metric)
    clf.fit(X_train, Y_train)
    print(clf.nodes.keys())
    print(" --> train acc: %s"%str(clf.score(X_train, Y_train)))
    print(" --> test acc.: %s"%str(clf.score(X_test, Y_test)))
    print("------- DONE -------\n")