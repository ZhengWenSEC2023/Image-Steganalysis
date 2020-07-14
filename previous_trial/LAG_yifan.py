# 2020.03.19v2
# label assistant regression
# modified from Yueru


import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors



class LAG():
    def __init__(self, learner, neigh, encode='onehot', num_clusters=[10, 10], alpha=5, par={}):
        self.learner = learner
        self.encode = encode
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.clus_labels = []
        self.centroid = []
        self.trained = False
        self.neigh = neigh
        self.knn_centroids = []
        self.knn_clus_labels = []


    def compute_target_(self, X, Y, batch_size):
        class_list = [1, 0]
        labels = np.zeros((X.shape[0]))
        self.clus_labels = np.zeros((np.sum(self.num_clusters),))
        self.centroid = np.zeros((np.sum(self.num_clusters), X.shape[1]))
        start = 0
        for i in range(len(class_list)):
            ID = class_list[i]
            if ID == 1:
                feature_train = X[Y == ID]

                if batch_size == None:
                    kmeans = KMeans(n_clusters=self.num_clusters[i], verbose=0, random_state=9, n_jobs=10).fit(
                        feature_train)
                else:
                    kmeans = MiniBatchKMeans(n_clusters=self.num_clusters[i], verbose=0, batch_size=batch_size,
                                             n_init=5).fit(feature_train)
                labels[Y == ID] = kmeans.labels_ + start
                self.clus_labels[start:start + self.num_clusters[i]] = ID
                self.centroid[start:start + self.num_clusters[i]] = kmeans.cluster_centers_
                start = start + self.num_clusters[i]

            if ID == 0:
                feature_train = X[Y == ID]

                if batch_size == None:
                    kmeans = KMeans(n_clusters=self.num_clusters[i], verbose=0, random_state=9, n_jobs=10).fit(
                        feature_train)
                else:
                    kmeans = MiniBatchKMeans(n_clusters=self.num_clusters[i], verbose=0, batch_size=batch_size,
                                             n_init=5).fit(feature_train)

                labels[Y == ID] = kmeans.labels_ + start
                self.clus_labels[start:start + self.num_clusters[i]] = ID
                self.centroid[start:start + self.num_clusters[i]] = kmeans.cluster_centers_

        return labels.astype('int32')

    def fit(self, X, Y, batch_size=None, knn=False):
        assert (len(self.num_clusters) >= np.unique(Y).shape[0]), "'len(num_cluster)' must larger than class number!"
        Yt = self.compute_target_(X, Y, batch_size=batch_size)

        if self.encode == 'distance':
            Yt_onehot = np.zeros((Yt.shape[0], self.clus_labels.shape[0]))
            for i in range(Yt.shape[0]):
                gt = Y[i].copy()
                dis = euclidean_distances(X[i].reshape(1, -1), self.centroid[self.clus_labels == gt]).reshape(-1)
                dis = dis / (np.min(dis) + 1e-15)
                p_dis = np.exp(-dis * self.alpha)
                p_dis = p_dis / np.sum(p_dis)
                Yt_onehot[i, self.clus_labels == gt] = p_dis

        elif self.encode == 'onehot':
            Yt_onehot = np.zeros((X.shape[0], np.unique(Yt).shape[0]))
            Yt_onehot[np.arange(Y.size), Yt] = 1

        else:
            print("       <Warning>        Using raw label for learner.")
            Yt_onehot = Yt

        if knn==True:
            self.neigh.fit(self.centroid)




        self.learner.fit(X, Yt_onehot)
        self.trained = True


    def predict(self, X):
        assert (self.trained == True), "Must call fit first!"
        return self.learner.predict(X)

    def predict_proba(self, X):
        assert (self.trained == True), "Must call fit first!"
        return self.learner.predict_proba(X)

    def score(self, X, Y, knn=False):
        assert (self.trained == True), "Must call fit first!"
        print("       <Warning>        Currently only support LLSR.")

        if knn == False:
            X = self.predict_proba(X)
            pred_labels = np.zeros((X.shape[0], len(np.unique(Y))))
            for km_i in range(len(np.unique(Y))):
                pred_labels[:, km_i] = np.sum(X[:, self.clus_labels == km_i], axis=1)
            pred_labels = np.argmax(pred_labels, axis=1)
            idx = (pred_labels == Y.reshape(-1))  # correct idx
            accuracy = np.count_nonzero(idx) / Y.shape[0]   # total accuracy


        if knn == True:

            idx_knn_centroids = self.neigh.kneighbors(X, return_distance=False)
            self.knn_centroids = self.centroid[idx_knn_centroids]
            self.knn_clus_labels = self.clus_labels[idx_knn_centroids]

            prob = np.zeros((X.shape[0], 2))
            knn_centroid_label = [[], []]
            pos_correct = 0
            neg_correct = 0

            for i in range(Y.shape[0]):
                gt = Y.reshape(-1)[i]
                knn_centroid_label[gt].append(len(self.knn_clus_labels[i][self.knn_clus_labels[i] == gt]) / len(self.knn_clus_labels[i]))


                dis = euclidean_distances(X[i].reshape(1, -1), self.knn_centroids[i]).reshape(-1)  # 10 distance
                dis = dis / (np.min(dis) + 1e-15)
                p_dis = np.exp(-dis * self.alpha)
                p_dis = p_dis / np.sum(p_dis)
                prob[i, 0] = np.sum(p_dis[self.knn_clus_labels[i] == 0])
                prob[i, 1] = np.sum(p_dis[self.knn_clus_labels[i] == 1])
                pred_label_i = np.argmax(prob[i])
                if gt == 1 and pred_label_i == 1:
                    pos_correct += 1
                if gt == 0 and pred_label_i == 0:
                    neg_correct += 1


            # pred_labels = np.argmax(prob, axis=1)
            #
            # pos_idx1 = np.where(pred_labels == 1)[0]
            # pos_idx2 = np.where(Y.reshape(-1) == 1)[0]
            pos_acc = pos_correct / np.sum(Y.reshape(-1)==1)
            #
            # neg_idx1 = np.where(pred_labels == 0)[0]
            # neg_idx2 = np.where(Y.reshape(-1) == 0)[0]
            neg_acc = neg_correct / np.sum(Y.reshape(-1)==0)
            accuracy = [neg_acc, pos_acc]

        return accuracy, knn_centroid_label


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from previous_trial.LLSR_yifan import LLSR

    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    print(" input feature shape: %s" % str(X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2, stratify=digits.target)

    clf = LAG(encode='distance', num_clusters=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], alpha=5, learner=LLSR(onehot=False), neigh=NearestNeighbors(10))
    clf.fit(X_train, y_train, knn=True)
    print(" --> train acc: %s" % str(clf.score(X_train, y_train, knn=True)))
    print(" --> test acc.: %s" % str(clf.score(X_test, y_test, knn=True)))
    print("------- DONE -------\n")