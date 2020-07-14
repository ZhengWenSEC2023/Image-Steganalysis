# v2020.03.19v2
# created by yifan

import numpy as np
from sklearn.metrics import accuracy_score
from numpy import inf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class LLSR():
    def __init__(self, onehot=True, normalize=False):
        self.onehot = onehot
        self.normalize = normalize
        self.coeff = []
        self.weight = []
        self.trained = False

    # weighted least square
    def fit(self, X, Y):
        if self.onehot == True:
            y = np.zeros((X.shape[0], np.unique(Y).shape[0]))
            y[np.arange(Y.size), Y] = 1
            Y = y.copy()



        bin_edge = np.linspace(-1, 1, 21, endpoint=True)
        counts, _ = np.histogram(Y, bins=bin_edge)

        freq = counts/Y.shape[0]

        idx = np.zeros((len(counts)+1), dtype=int)  # 21
        for i in range(len(counts)+1):
            idx[i] = np.sum(counts[:i])

        W = np.zeros((Y.shape[0]))
        for i in range(len(idx)-1):
            for j in range(idx[i], idx[i+1]):
                W[j] = 1/max(freq[i], 0.001)


        # X_pos = X[Y>0]
        # X_neg = X[Y<0]
        # Y_pos = Y[Y>0]
        # Y_neg = Y[Y<0]
        #
        # X_reorder = np.concatenate((X_pos, X_neg), axis=0)
        # Y_reorder = np.concatenate((Y_pos, Y_neg), axis=0)
        #
        # km_p = KMeans(n_clusters=10).fit(Y_pos.reshape(Y_pos.shape[0], 1))
        # km_n = KMeans(n_clusters=10).fit(Y_neg.reshape(Y_neg.shape[0], 1))
        #
        # freq_p = np.zeros((10))
        # freq_n = np.zeros((10))
        #
        # pos_clus_labels = km_p.labels_
        # neg_clus_labels = km_n.labels_
        #
        # for i in range(10):
        #     freq_p[i] = len(Y_pos[pos_clus_labels == i]) / len(Y_pos)
        #     freq_n[i] = len(Y_neg[neg_clus_labels == i]) / len(Y_neg)
        #
        # W = np.zeros((Y.shape[0]))
        #
        # for i in range(Y_pos.shape[0]):
        #     label = km_p.predict(Y_pos[i].reshape(1, 1))
        #     W[i] = 1/(freq_p[label] + 0.001)
        # for j in range(Y_neg.shape[0]):
        #     label = km_n.predict(Y_neg[j].reshape(1, 1))
        #     W[Y_pos.shape[0] + j] = 1/(freq_n[label] + 0.001)






        # W = np.ones((Y.shape[0]))

        self.weight = W

        WX = np.zeros((X.shape))
        WY = np.zeros((Y.shape))
        for i in range(X.shape[0]):
            WX[i] = W[i] * np.array(X[i])
            WY[i] = W[i] * np.array(Y[i])

        # if Y.shape[0] == 986864:
        #     plt.figure()
        #     edge = np.linspace(np.min(WY), np.max(WY), 41, endpoint=True)
        #     n_p, bins_p, patches_p = plt.hist(x=WY, bins=edge, color='b', rwidth=0.9, density=True, label="spliced pixels")
        #
        #     plt.grid(axis='y')
        #     plt.xlabel('Value')
        #     plt.ylabel('Frequency')
        #     plt.title('histogram of weighted training Y')
        #     # plt.legend()
        #     plt.savefig('/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/multi_resolution/features/' + "1-2_resolution/ssl/histogram_1-2_lab_weighted_train_Y.png")

        A = np.ones((WX.shape[0], 1))
        WX = np.concatenate((A, WX), axis=1)


        self.coeff = np.matmul(np.linalg.pinv(WX), WY)
        self.trained = True

        return self

    # def predict(self, X):
    #     assert (self.trained == True), "Must call fit first!"
    #     X = self.predict_proba(X)
    #     return np.argmax(X, axis=1)

    def predict_proba(self, X):
        assert (self.trained == True), "Must call fit first!"
        A = np.ones((X.shape[0], 1))
        X = np.concatenate((A, X), axis=1)
        pred = np.matmul(X, self.coeff)
        if self.normalize == True:
            pred = (pred - np.min(pred, axis=1, keepdims=True)) / np.sum(
                (pred - np.min(pred, axis=1, keepdims=True) + 1e-15), axis=1, keepdims=True)
        return pred

    # def score(self, X, Y):
    #     assert (self.trained == True), "Must call fit first!"
    #     pred = self.predict(X)
    #     return accuracy_score(Y, pred)

    def mse(self, X, Y):
        assert (self.trained == True), "Must call fit first"
        pred = self.predict_proba(X)
        return mean_squared_error(Y, pred, sample_weight=self.weight)




if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    print(" input feature shape: %s" % str(X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2, stratify=digits.target)

    clf = LLSR(onehot=True, normalize=True)
    clf.fit(X_train, y_train)
    print(" --> train acc: %s" % str(clf.score(X_train, y_train)))
    print(" --> test acc: %s" % str(clf.score(X_test, y_test)))
    print("------- DONE -------\n")