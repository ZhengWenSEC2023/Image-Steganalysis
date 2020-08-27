# 2020.04.14
from final_pixel_pipeline.cwSaab import cwSaab
import numpy as np

class Pixelhop2(cwSaab):
    def __init__(self, depth=1, TH1=0.01, TH2=0.001, SaabArgs=None, shrinkArgs=None, concatArg=None, splitMode=2):
        super().__init__(depth=depth, energyTH=TH1, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg={'func':lambda X, concatArg: X}, splitMode=splitMode)
        self.TH1 = TH1
        self.TH2 = TH2
        self.concatArg = concatArg

    def select_(self, X):
        for i in range(self.depth):
            X[i] = X[i][:, :, :, self.Energy[i] >= self.TH2]
        # for i in range(self.depth-1):
        #     # X[i] = X[i][:, :, :, self.Energy[i] >= self.TH2]
        #     idx = np.logical_and(np.array(self.splitidx[i]) == False, self.Energy[i] >= self.TH2)
        #     X[i] = X[i][:, :, :, idx]
        # X[self.depth-1] = X[self.depth-1][:, :, :, self.Energy[self.depth-1] >= self.TH2]
        return X

    def construct_count(self):
        self.counts = []
        for i in range(self.depth):
            count = []
            for k in range(1, self.Energy[i].shape[-1] // 25 + 1):
                cur_count = np.sum(self.Energy[i][(k - 1) * 25: k * 25] >= self.TH2)
                count.append(cur_count)
            self.counts.append(count)
        for i in range(len(self.counts)):
            while 0 in self.counts[i]:
                self.counts[i].remove(0)
            self.counts[i].insert(0, 0)
        for i in range(len(self.counts)):
            self.counts[i] = np.cumsum(self.counts[i])
        for i in range(1, len(self.counts)):
            self.counts[i] += self.counts[i - 1][-1]

    def fit(self, X):
        super().fit(X)
        return self

    def transform(self, X):
        X = super().transform(X)
        X = self.select_(X)
        return self.concatArg['func'](X, self.concatArg)

# if __name__ == "__main__":
#     # example useage
#     from sklearn import datasets
#     from skimage.util import view_as_windows
#     import cv2
#     import time
#     # example callback function for collecting patches and its inverse
#     def Shrink(X, shrinkArg, max_pooling):
#         win = shrinkArg['win']
#         X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
#         return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
#
#     # example callback function for how to concate features from different hops
#     def Concat(X, concatArg):
#         return X
#
#     # read data
#     print(" > This is a test example: ")
#     digits = datasets.load_digits()
#     X = digits.images.reshape((len(digits.images), 8, 8, 1))
#     print(" input feature shape: %s"%str(X.shape))
#
#     # set args
#     SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None},
#                 {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None}]
#     shrinkArgs = [{'func':Shrink, 'win':3},
#                 {'func': Shrink, 'win':3}]
#     concatArg = {'func':Concat}
#
#     print(" --> test inv")
#     print(" -----> depth=1")
#     p2 = Pixelhop2(depth=1, TH1=0.01, TH2=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
#     output = p2.fit(X)
