# 2020.05.19

# modified by Yao Zhu


import numpy as np
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from previous_trial.mylearner import myLearner


class HierNode():
    def __init__(self, learner, num_class, num_cluster, metric, isleaf=False, id='R'):
        self.learner = myLearner(learner=learner, num_class=num_class)
        self.kmeans = KMeans(n_clusters=num_cluster)
        self.num_cluster = num_cluster
        self.metric = metric
        self.isleaf = isleaf
        self.id = id

    def metric_(self, X, Y):
        if 'func' in self.metric.keys():
            return self.metric['func'](X, Y, self.metric)
        if X.shape[0] < self.num_cluster * self.metric['min_num_sample']:
            return True
        return False

    def fit(self, X, Y):
        self.kmeans.fit(X)
        if self.metric_(X, Y) == True:
            self.isleaf = True
        if self.isleaf == True:
            self.learner.fit(X, Y)
        return self

    def predict(self, X):
        if self.isleaf == True:
            try:
                prob = self.learner.predict_proba(X)
            except:
                prob = self.learner.predict(X)
            return prob
        else:
            return self.kmeans.predict(X)





class HierKmeans():
    def __init__(self, depth, learner, num_cluster, metric):
        self.nodes = {}
        self.depth = depth
        self.learner = learner
        self.num_cluster = num_cluster
        self.metric = metric
        self.num_class = -1
        self.trained = False

    def fit(self, X, Y):
        self.num_class = len(np.unique(Y))
        tmp_data = [{'X': X, 'Y': Y, 'id': 'R'}]
        for i in range(self.depth):
            tmp = []
            for j in range(len(tmp_data)):
                if tmp_data[j]['X'].shape[0] == 0:
                    continue
                # tmp_node = HierNode_fancy(learner=self.learner,
                # tmp_node = HierNode_query(learner=self.learner,
                tmp_node = HierNode(learner=self.learner,
                                    num_class=self.num_class,
                                    num_cluster=self.num_cluster,
                                    metric=self.metric,
                                    isleaf=(i == self.depth - 1),
                                    id=tmp_data[j]['id'])
                tmp_node.fit(tmp_data[j]['X'], tmp_data[j]['Y'])
                label = tmp_node.predict(tmp_data[j]['X'])
                self.nodes[tmp_data[j]['id']] = copy.deepcopy(tmp_node)
                if tmp_node.isleaf == True:
                    continue
                for k in range(self.num_cluster):
                    idx = (label == k)
                    tmp.append(
                        {'X': tmp_data[j]['X'][idx], 'Y': tmp_data[j]['Y'][idx], 'id': tmp_data[j]['id'] + str(k)})
            if len(tmp) == 0 and i != self.depth - 1:
                print("       <Warning> depth %s not achieved, actual depth %s" % (str(self.depth), str(i + 1)))
                self.depth = i
                break
            tmp_data = tmp
        self.trained = True

    def predict_proba(self, X):
        assert (self.trained == True), "Must call fit first!"
        tmp_pred = []
        tmp_data = [{'X': X, 'idx': np.arange(0, X.shape[0]), 'id': 'R'}]
        for i in range(self.depth):
            tmp = []
            for j in range(len(tmp_data)):
                if tmp_data[j]['X'].shape[0] == 0:
                    continue
                if self.nodes[tmp_data[j]['id']].isleaf == True:
                    prob = self.nodes[tmp_data[j]['id']].predict(tmp_data[j]['X'])
                    tmp_pred.append({'prob': prob.reshape(prob.shape[0], -1), 'idx': tmp_data[j]['idx']})
                    continue
                label = self.nodes[tmp_data[j]['id']].predict(tmp_data[j]['X'])
                for k in range(self.num_cluster):
                    idx = (label == k)
                    tmp.append({'X': tmp_data[j]['X'][idx],
                                'idx': tmp_data[j]['idx'][idx],
                                'id': tmp_data[j]['id'] + str(k)})
            tmp_data = tmp
        prob, idx = [], []
        for i in range(len(tmp_pred)):
            prob.append(tmp_pred[i]['prob'])
            idx.append(tmp_pred[i]['idx'])
        prob = np.concatenate(prob, axis=0)
        idx = np.concatenate(idx, axis=0)
        idx = np.argsort(idx)
        return prob[idx]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, Y):
        return accuracy_score(Y, self.predict(X))


if __name__ == "__main__":
    from sklearn.svm import SVC
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    print(" input feature shape: %s" % str(X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2, stratify=digits.target)

    metric = {'min_num_sample': 10,
              'purity': 0.9,
              'min_percent_query': 0.05,
              'query_trial_percent': 0.2}
    clf = HierKmeans(depth=3, learner=SVC(gamma='scale', probability=True), num_cluster=3, metric=metric)
    clf.fit(X_train, y_train)
    print(clf.nodes.keys())
    print(" --> train acc: %s" % str(clf.score(X_train, y_train)))
    print(" --> test acc.: %s" % str(clf.score(X_test, y_test)))
    print("------- DONE -------\n")