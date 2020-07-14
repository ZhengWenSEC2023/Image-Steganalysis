# 2020.04.23
# Hierachical Kmeans
# author: yifan
# modified by Yao Zhu


import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist


def Init_LBG(X, sep_num=2):
    c1 = np.mean(X, axis=0).reshape(1, -1)
    st = np.std(X, axis=0).reshape(1, -1)
    c2 = c1 + st
    new_centroids = np.concatenate((c1, c2), axis=0)
    return new_centroids


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
        inertia = {}
        # LBG
        initial_centroids = Init_LBG(X)
        km_LBG = MiniBatchKMeans(n_clusters=2, batch_size=1000, init=initial_centroids, n_init=1).fit(X)

        inertia[km_LBG] = km_LBG.inertia_

        # k-means++
        km_plus = MiniBatchKMeans(n_clusters=2, batch_size=1000, init='k-means++', n_init=20).fit(X)
        inertia[km_plus] = km_plus.inertia_

        # random sample
        km_random = MiniBatchKMeans(n_clusters=2, batch_size=1000, init='random', n_init=20).fit(X)
        inertia[km_random] = km_random.inertia_

        # mean of different class
        init_mean = np.zeros((2, X.shape[-1]))

        init_mean[0] = np.mean(X[Y > 0.5], axis=0)
        init_mean[1] = np.mean(X[Y < 0.5], axis=0)

        km_mean = MiniBatchKMeans(n_clusters=2, batch_size=1000, init=init_mean, n_init=1).fit(X)
        inertia[km_mean] = km_mean.inertia_

        min_value = 10e30
        for key, value in inertia.items():
            if value < min_value:
                min_value = value
                self.kmeans = key

        print(self.kmeans)

    def fit(self, X, Y):
        self.multi_trial(X, Y)


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

    def source_mse_decrease(self, X, Y):
        clus_label = self.predict(X)

        self.source_mse_parent = np.mean(np.var(X, axis=0))

        self.gtvar_parent = np.var(Y) * len(Y)  # total variance, not average variance

        sum_child_mse = 0.0

        for k in range(self.num_cluster):
            clus_data = X[clus_label == k]
            clus_gt = Y[clus_label == k]

            self.gtvar_child[k] = np.var(clus_gt) * len(clus_gt)  # total variance, not average variance
            self.source_mse_child[k] = np.mean(np.var(clus_data, axis=0))

            sum_child_mse += self.source_mse_child[k] * len(clus_data) / len(X)

        return self.source_mse_parent - sum_child_mse

def cost_function(X, Y, centroid):
    sse = np.mean(np.square(X - centroid))
    purity = np.max((np.sum(Y != 0), np.sum(Y == 0))) / len(Y)
    number = len(X)
    print(np.sqrt(sse), 100 * purity, number / 1000)
    return np.sqrt(sse) + 100 * purity + number / 1000

class HierKmeans():
    def __init__(self, depth, min_sample_num, num_cluster, metric, istop=True, topid=None, topsampleidx=None,
                 topgtvar=None, topmse=None):
        self.nodes = {}
        self.depth = depth
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
        self.cost_func = []

    def fit(self, X, Y):
        # self.num_class = 2
        X = X.reshape(-1, X.shape[-1])
        if self.istop == True:
            tmp_data = [{'X': X, 'Y': Y, 'id': 'R', 'mse': 0}]  # leaf nodes candidates
        else:
            tmp_data = [{'X': X, 'Y': Y, 'id': self.topid, 'mse': self.topmse, 'sample_idx': self.topsampleidx,
                         'gtvar': self.topgtvar}]

        source_mse_decrease = []
        gtvar = []
        global_gtvar = []
        global_source_mse = []

        global_cost_function = []

        for i in range(self.depth):  # splitting的次数
            tmp = []
            if i == 0:  # root node
                self.nodes['R'] = []

                tmp_node = HierNode(
                    # num_class=self.num_class,
                    num_cluster=self.num_cluster,
                    metric=self.metric,
                    isleaf=(i == self.depth - 1),
                    id=tmp_data[i]['id'])
                tmp_node.fit(tmp_data[i]['X'], tmp_data[i]['Y'])

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
                             'sample_idx': np.where(idx == True)[0],
                             'centroid': tmp_node.kmeans.cluster_centers_[k]})

                        self.nodes[tmp_data[i]['id'] + str(k)] = {'X': tmp_data[i]['X'][idx],
                                                                  'Y': tmp_data[i]['Y'][idx],
                                                                  'mse': tmp_node.source_mse_child[k],
                                                                  'gtvar': tmp_node.gtvar_child[k],
                                                                  'sample_idx': np.where(idx == True)[0],
                                                                  'centroid': tmp_node.kmeans.cluster_centers_[k]
                                                                  }
                        gtvar.append(tmp_node.gtvar_child[k])
                else:
                    print(tmp_node.source_mse_parent, self.topmse)
                    print(tmp_node.gtvar_parent, self.topgtvar)
                    # assert tmp_node.source_mse_parent == self.topmse, "mse start of bottom of tree does not match top leaf node"
                    # assert tmp_node.gtvar_parent == self.topgtvar, "gt var start of bottom of tree does not match top leaf node"

                    for k in range(self.num_cluster):
                        idx = (label == k)
                        tmp.append(
                            {'X': tmp_data[i]['X'][idx],
                             'Y': tmp_data[i]['Y'][idx],
                             'id': tmp_data[i]['id'] + str(k),
                             'mse': tmp_node.source_mse_child[k],
                             'gtvar': tmp_node.gtvar_child[k],
                             'sample_idx': tmp_data[i]['sample_idx'][idx],
                             'centroid': tmp_node.kmeans.cluster_centers_[k]
                             })

                        self.nodes[tmp_data[i]['id'] + str(k)] = {'X': tmp_data[i]['X'][idx],
                                                                  'Y': tmp_data[i]['Y'][idx],
                                                                  'mse': tmp_node.source_mse_child[k],
                                                                  'gtvar': tmp_node.gtvar_child[k],
                                                                  'sample_idx': tmp_data[i]['sample_idx'][idx],
                                                                  'centroid': tmp_node.kmeans.cluster_centers_[k]}
                        gtvar.append(tmp_node.gtvar_child[k])

                tmp_data = tmp

            if i != 0:
                # find which node to split
                cost_func = np.zeros((len(tmp_data)))
                for j in range(len(tmp_data)):
                    # if tmp_data[j]['X'].shape[0] > self.min_sample_num:
                    cost_func[j] = cost_function(tmp_data[j]['X'], tmp_data[j]['Y'], tmp_data[j]["centroid"])
                idx_to_split = int(np.argmax(cost_func))
                self.cost_func.append(np.sum(cost_func))

                key = tmp_data[idx_to_split]['id']
                self.nodes[key] = []  # 清空要分的node里的sample

                tmp_node = HierNode(
                    # num_class=self.num_class,
                    num_cluster=self.num_cluster,
                    metric=self.metric,
                    isleaf=(i == self.depth - 1),
                    id=tmp_data[idx_to_split]['id'])

                tmp_node.fit(tmp_data[idx_to_split]['X'], tmp_data[idx_to_split]['Y'])
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
                         'sample_idx': sample_idx,
                         'centroid': tmp_node.kmeans.cluster_centers_[k]
                         })  # be careful! this idx is based on its parent node!

                    self.nodes[tmp_data[idx_to_split]['id'] + str(k)] = {'X': tmp_data[idx_to_split]['X'][idx],
                                                                         'Y': tmp_data[idx_to_split]['Y'][idx],
                                                                         'mse': tmp_node.source_mse_child[k],
                                                                         'gtvar': tmp_node.gtvar_child[k],
                                                                         'sample_idx': sample_idx,
                                                                         'centroid': tmp_node.kmeans.cluster_centers_[
                                                                             k]}
                    gtvar.append(tmp_node.gtvar_child[k])

                tmp_data = tmp

                # calculate global source mse in this splitting
                global_source_mse_i = 0.0
                for i in range(len(tmp_data)):
                    global_source_mse_i += tmp_data[i]['mse'] * (len(tmp_data[i]['X']) / len(X))

                global_source_mse.append(global_source_mse_i)

            # print("decrease of MSE:", source_mse_decrease)
            print('\n')
            print("gt variance:", gtvar)
            global_gtvar.append(np.sum(gtvar))
            print("global gt var:", global_gtvar)
            print("global source mse:", global_source_mse)

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

            if len(tmp) == 0 and i != self.depth - 1:
                print("       <Warning> depth %s not achieved, actual depth %s" % (str(self.depth), str(i + 1)))
                self.depth = i
                break

        self.trained = True
        print(self.cost_func)

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
        #     # self.leaf_learner.append(
        #     #     RandomForestRegressor(n_estimators=10, min_samples_leaf=60, min_samples_split=100).fit(leaf_nodes_X[k],
        #     #                                                                                            leaf_nodes_Y[k]))

    def predict_proba(self, X, position):
        """
        Need positions of X to find out where the stego pixels go
        position: a list consist of 3-elements tuples with the same length with X
        X: data matrix

        TO BE REFINED AS A TREE STRUCTURE!!!

        """

        assert (self.trained == True), "Must call fit first!"
        centroids = []
        leafs = {}
        keys = []
        key_list = list(self.nodes.keys())

        for i in range(len(key_list)):
            key = key_list[i]
            if self.nodes[key] != []:
                centroids.append(self.nodes[key]['centroid'])
                keys.append(key)
                leafs[key] = {
                    'X': [],
                    'pos': []
                }
        centroids = np.array(centroids)
        for i in range(len(X)):
            X_temp = np.tile(X[i], (centroids.shape[0], 1))
            dist = cdist(X_temp, centroids)
            index = np.argmin(dist)
            leafs[keys[index]]['X'].append(X[i])
            leafs[keys[index]]['pos'].append(position[i])

        return leafs

    def score(self, X, Y):
        Y_pred = self.predict_proba(X)

        return mean_squared_error(Y, Y_pred)


def findDiffLeafs(leafsA, leafsB, diff_maps):
    """
    find the the leaf where the stego signals go.
    return changed purity and id
    leafsA is the original leafs,
    leafsB is the stego leafs.

    e.g. if run this function with original images and stego images,
         the purity of original images will be 100%
         the purity of stego images will vary with payload.
         the id of changed node will be also returned.
    """


    key_list = list(leafsA.keys())
    unchanged_leaf = []
    changed_leaf = []
    print('total lists:', len(key_list))
    for i in range(len(key_list)):
        print(i)
        key = key_list[i]
        pos_A = leafsA[key]['pos']
        pos_B = leafsB[key]['pos']
        pos_A = np.array(pos_A).tolist()
        pos_B = np.array(pos_B).tolist()
        unique_in_A = []
        for pos_temp in pos_A:
            if pos_temp not in pos_B:
                unique_in_A.append(pos_temp)
        unique_in_B = []
        for pos_temp in pos_B:
            if pos_temp not in pos_A:
                unique_in_B.append(pos_temp)
        if not unique_in_A and not unique_in_B:
            unchanged_leaf.append(key)
        else:
            tmp_dict = {}
            tmp_dict['ID'] = key
            tmp_dict['pos_changed'] = np.sum(len(unique_in_A) + len(unique_in_B))
            num_stego = 0
            for num, ii, jj in unique_in_B:
                if diff_maps[num, ii, jj] != 0:
                    num_stego += 1

            tmp_dict[r'purity % changed'] = num_stego / len(pos_A)
            tmp_dict[r'purity # changed'] = num_stego
            changed_leaf.append(tmp_dict)
    return changed_leaf, unchanged_leaf


if __name__ == "__main__":
    from previous_trial.config_img_stag import config

    save_dir_matFiles = config['save_dir_matFiles']
    window_rad = config['wind_rad']
    thres_bin_edges = config['thres_bin_edges']
    ori_image_dir = config['ori_image_dir']
    stego_image_dir = config['stego_image_dir']
    context_dir = config['context_dir']
    train_ori_context_name = config['train_ori_context_name']
    weight_root_name = config['weight_root_name']
    train_pos_target_name = config['train_pos_target_name']
    train_neg_target_name = config['train_neg_target_name']
    train_pos_contxt_name = config['train_pos_contxt_name']
    train_neg_contxt_name = config['train_neg_contxt_name']
    test_ori_contxt_name = config['test_ori_contxt_name']
    test_contxt_name = config['test_contxt_name']
    regressor_dir = config['regressor_dir']
    regressor_name = config['regressor_name']
    plot_dir = config['plot_dir']
    one_hop_stp_lst_name = config['one_hop_stp_lst_name']

    """
    all target values are comes from original image
    """
    used_nums = 2000000
    context_unchanged = np.load('week5_context_unchanged.npy')
    context_changed = np.load('week5_context_changed.npy')

    metric = {'min_num_sample': 200, 'purity': 0.9}

    context = np.concatenate((context_changed, context_unchanged), axis=0)
    train_target = np.concatenate((np.ones(context_changed.shape[0]), np.zeros(context_unchanged.shape[0])), axis=0)

    idx = np.random.permutation(train_target.shape[0])
    context = context[idx][:used_nums]
    train_target = train_target[idx][:used_nums]

    # clf_top = HierKmeans(depth=15, min_sample_num=200000//5, num_cluster=2, metric=metric, istop=True)
    clf_top = HierKmeans(depth=100, min_sample_num=10000, num_cluster=2, metric=metric, istop=True)
    clf_top.fit(context, train_target)
    print(clf_top.nodes.keys())

    # leafs_cover = clf_top.predict_proba(context_top, pos_total)
    # leafs_stego = clf_top.predict_proba(context_steg, pos_total)
    # changed_leaf, unchanged_leaf = findDiffLeafs(leafs_cover, leafs_stego, diffs)
    #
    # pkl = open('trained_tree.pkl', 'wb')
    # pickle.dump(clf_top, pkl)
    # pkl.close()
    #
    # pkl = open('original_leafs.pkl', 'wb')
    # pickle.dump(leafs_cover, pkl)
    # pkl.close()
    #
    # pkl = open('stego_leafs_0.05.pkl', 'wb')
    # pickle.dump(leafs_stego, pkl)
    # pkl.close()
    #
    # pkl = open('changed_leaf_0.05.pkl', 'wb')
    # pickle.dump(changed_leaf, pkl)
    # pkl.close()
    #
    # leaf_nodes_id = []
    # leaf_nodes_mse = []
    # leaf_nodes_gt_purity = []
    # leaf_nodes_X_idx = []
    #
    # for node_id in clf_top.nodes.keys():
    #     if clf_top.nodes[node_id] != []:
    #         print("node id and leaf size:", node_id, clf_top.nodes[node_id]['X'].shape[0])
    #
    #         leaf_nodes_id.append(node_id)
    #         leaf_nodes_mse.append(clf_top.nodes[node_id]['mse'])
    #         leaf_nodes_gt_purity.append(clf_top.nodes[node_id]['gt_purity'])
    #         leaf_nodes_X_idx.append(clf_top.nodes[node_id]['sample_idx'])
    #
    # # check sample_idx for all leaf nodes, make sure no overlap:
    # leaf_idx_set = [[] for i in range(len(leaf_nodes_X_idx))]
    # whole_leaf_idx_check = set()
    #
    # for i in range(len(leaf_nodes_X_idx)):
    #     if len(set(leaf_nodes_X_idx[i])) == len(leaf_nodes_X_idx[i]):
    #         set_i = set(leaf_nodes_X_idx[i])
    #
    #         whole_leaf_idx_check = whole_leaf_idx_check.union(set_i)
    #
    # if len(whole_leaf_idx_check) == context_4.shape[0] * context_4.shape[1] * context_4.shape[2]:
    #     print("!!!! Top of tree constructed !!!!")
    #
    # del leaf_idx_set, whole_leaf_idx_check
    # # del clf_top
    #
    # # check variance in top tree leaf nodes
    # top_num_samples_small_var = 0
    # top_leaf_variance = []
    # for k in range(len(leaf_nodes_X_idx)):
    #     top_leaf_idx = leaf_nodes_X_idx[k]
    #     top_leaf_target = train_target[top_leaf_idx]
    #     top_leaf_variance.append(np.var(top_leaf_target))
    #     if np.var(top_leaf_target) < init_var:
    #         top_num_samples_small_var += len(top_leaf_target)
    #
    # top_leaf_variance = np.array(top_leaf_variance)
    #
    # print(np.sum(top_leaf_variance > init_var), np.sum(top_leaf_variance < init_var))
    # print(top_num_samples_small_var)
    #
    # plt.figure(0)
    # n_p, bins_p, patches_p = plt.hist(x=top_leaf_variance, bins=10, color='b',
    #                                   alpha=0.7, rwidth=0.85)
    # # n_n, bins_n, patches_n = plt.hist(x=top_leaf_variance[top_leaf_variance < init_var], bins='auto', color='y',
    # #                                   alpha=0.7, rwidth=0.85)
    #
    # plt.grid(axis='y', alpha=0.75)
    # plt.title('histogram of leaf node GT variance')
    #
    # plt.savefig(save_dir_matFiles + "original_resolution/context/leaf_target_hist/HierKM_topleaf_GT_purity.png")
    # plt.close(0)
    #
    # # %% use concatenation of context 1,2,3 to finish bottom of tree
    #
    # context_bottom = np.concatenate((context_2, context_3), axis=3)
    # context_bottom = context_bottom.reshape(-1, context_bottom.shape[-1])
    #
    # bottom_leaf_idx = []
    # for k in range(len(leaf_nodes_X_idx)):
    #     start_node_id = leaf_nodes_id[k]
    #     start_node_mse = leaf_nodes_mse[k]
    #     start_node_gt_purity = leaf_nodes_gt_purity[k]
    #
    #     leaf_samp_idx = leaf_nodes_X_idx[k]
    #     leaf_samp_context123 = context_bottom[leaf_samp_idx]
    #     leaf_samp_gt = train_target[leaf_samp_idx]
    #
    #     print("--- start training sub-tree for leaf node {} ---".format(start_node_id))
    #
    #     metric_bot = {'min_num_sample': 500, 'purity': 0.9}
    #     clf_bot = HierKmeans(depth=15, min_sample_num=3000, num_cluster=2, metric=metric_bot, istop=False,
    #                          topid=start_node_id, topmse=start_node_mse, topgt_purity=start_node_gt_purity,
    #                          topsampleidx=leaf_samp_idx)
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
    # print(np.sum(variance > init_var), np.sum(variance < init_var))
    # print(num_samples_small_var)
    #
    # sorted_var_idx = np.argsort(variance)
    # sorted_var = variance[sorted_var_idx]
    #
    # # plot the histogram of variance in leaf nodes
    #
    # variance = np.array(variance)
    # print(np.sum(variance > init_var), np.sum(variance < init_var))
    # bin_edge = np.linspace(0, 0.7, 41, endpoint=True)
    # plt.figure(0)
    # n_p, bins_p, patches_p = plt.hist(x=variance, bins='auto', color='b',
    #                                   alpha=0.7, rwidth=0.85)
    #
    # plt.grid(axis='y', alpha=0.75)
    # plt.title('histogram of leaf node GT variance')
    #
    # plt.savefig(save_dir_matFiles + "original_resolution/context/leaf_target_hist/HierKM_leafnodes_gt_purity.png")
    # plt.close(0)
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
    # print("------- DONE -------\n")
    #
    # # # direct apply regular kmeans
    # # direct_input = np.concatenate((context_1, context_2, context_3, context_4), axis=3)
    # # direct_input = direct_input.reshape(-1, direct_input.shape[-1])
    # #
    # # km_direct = MiniBatchKMeans(n_clusters=256, batch_size=1000).fit(direct_input)
    # # init_var = np.var(train_target)
    # # print("initial variance:", init_var)
    # #
    # # label = km_direct.labels_
    # # variance = np.zeros((256))
    # # num_small_variance = 0
    # # for k in range(256):
    # #     idx = (label == k)
    # #     leaf_gt = train_target[idx]
    # #     var = np.var(leaf_gt)
    # #     variance[k] = var
    # #     if var < init_var:
    # #         num_small_variance += len(leaf_gt)
    # #
    # #     print("node {} length".format(k), len(leaf_gt))
    # #     print("        variance", var)
    # #
    # #
    # #     weak = []
    # #     for i in range(len(leaf_gt)):
    # #         if leaf_gt[i] > -0.2 and leaf_gt[i] < 0.2:
    # #             weak.append(leaf_gt[i])
    # #
    # #     plt.figure(0)
    # #     bin_edge = np.linspace(-1, 1, 41, endpoint=True)
    # #     n_p, bins_p, patches_p = plt.hist(x=leaf_gt[leaf_gt > 0.2], bins=bin_edge, color='b',
    # #                                       alpha=0.7, rwidth=0.85, label="spliced pixels")
    # #     n_n, bins_n, patches_n = plt.hist(x=leaf_gt[leaf_gt < -0.2], bins=bin_edge, color='y',
    # #                                       alpha=0.7, rwidth=0.85, label="authentic pixels")
    # #     n_w, bins_w, patches_w = plt.hist(x=weak, bins=bin_edge, color='r',
    # #                                       alpha=0.7, rwidth=0.85, label="background pixels")
    # #
    # #     plt.grid(axis='y', alpha=0.75)
    # #     plt.title('histogram of regular km train target in leaf node {}, variance = {}'.format(k, var))
    # #     plt.legend()
    # #
    # #     plt.savefig(save_dir_matFiles + "original_resolution/context/leaf_target_hist/regkm_leaf_hist_{}.png".format(k))
    # #     plt.close(0)
    # #
    # #
    # # # plot the histogram of variance in leaf nodes
    # # print(num_small_variance)
    # # variance = np.array(variance)
    # # print(np.sum(variance > init_var), np.sum(variance < init_var))
    # # bin_edge = np.linspace(0, 0.5, 41, endpoint=True)
    # # plt.figure(0)
    # # n_p, bins_p, patches_p = plt.hist(x=variance, bins=50, color='b',
    # #                                   alpha=0.7, rwidth=0.85)
    # #
    # # plt.grid(axis='y', alpha=0.75)
    # # plt.title('histogram of leaf node GT variance')
    # #
    # # plt.savefig(save_dir_matFiles + "original_resolution/context/leaf_target_hist/reg_leafnodes_GTvar.png")
    # # plt.close(0)
    # #
