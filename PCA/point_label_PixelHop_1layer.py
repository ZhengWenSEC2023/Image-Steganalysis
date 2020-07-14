import numpy as np
from PCA.saab import Saab
from PCA.PixelHop2 import readImgToNumpy_pairs



class PointLabelcwSaab:
    def __init__(self, depth=1, energyTH=0.01, SaabArgs=None, shrinkArgs=None):
        self.par = {}
        assert (depth > 0), "'depth' must > 0!"
        self.depth = int(depth)
        self.energyTH = energyTH
        assert (SaabArgs is not None), "Need parameter 'SaabArgs'!"
        self.SaabArgs = SaabArgs
        assert (shrinkArgs is not None), "Need parameter 'shrinkArgs'!"
        self.shrinkArgs = shrinkArgs
        self.Energy = []
        self.trained = False
        self.split = False
        if depth > np.min([len(SaabArgs), len(shrinkArgs)]):
            self.depth = np.min([len(SaabArgs), len(shrinkArgs)])
            print("       <WARNING> Too few 'SaabArgs/shrinkArgs' to get depth %s, actual depth: %s" % (
                str(depth), str(self.depth)))

    def SaabTransform(self, X, saab, train, layer, diffs, label):
        """
        Label 0: unchanged pts
        Label 1: changed pts
        """

        shrinkArg, SaabArg = self.shrinkArgs[layer], self.SaabArgs[layer]
        assert ('func' in shrinkArg.keys()), "shrinkArg must contain key 'func'!"

        X = shrinkArg['func'](X, shrinkArg)
        S = list(X.shape)

        new_X = []
        for k in range(S[0]):
            curr_X = X[k]
            curr_diff = np.squeeze(diffs[k])
            new_X.extend(curr_X[np.where(curr_diff != 0)])
        new_X = np.array(new_X)
        new_X = new_X.reshape(-1, S[-1])
        if SaabArg['num_AC_kernels'] != -1:
            S[-1] = SaabArg['num_AC_kernels']
        if train:
            saab = Saab(num_kernels=SaabArg['num_AC_kernels'], useDC=SaabArg['useDC'], needBias=SaabArg['needBias'])
            saab.fit(new_X)
        X = X.reshape(-1, S[-1])
        transformed, dc = saab.transform(X)
        transformed = transformed.reshape(S)
        return saab, transformed, dc

    def cwSaab_1_layer(self, X, train, diffs, label):
        if train:
            saab_cur = []
        else:
            saab_cur = self.par['Layer' + str(0)]
        transformed, eng, DC = [], [], []
        if self.SaabArgs[0]['cw']:
            S = list(X.shape)
            S[-1] = 1
            X = np.moveaxis(X, -1, 0)
            for i in range(X.shape[0]):
                X_tmp = X[i].reshape(S)
                if train:
                    saab, tmp_transformed, dc = self.SaabTransform(X_tmp, saab=None, train=True, layer=0, diffs=diffs, label=label)
                    saab_cur.append(saab)
                    eng.append(saab.Energy)
                else:
                    if len(saab_cur) == i:
                        break
                    _, tmp_transformed, dc = self.SaabTransform(X_tmp, saab=saab_cur[i], train=False, layer=0, diffs=diffs, label=label)
                transformed.append(tmp_transformed)
                DC.append(dc)
            transformed = np.concatenate(transformed, axis=-1)
        else:
            if train:
                saab, transformed, dc = self.SaabTransform(X, saab=None, train=True, layer=0, diffs=diffs, label=label)
                saab_cur.append(saab)
                eng.append(saab.Energy)
            else:
                _, transformed, dc = self.SaabTransform(X, saab=saab_cur[0], train=False, layer=0, diffs=diffs, label=label)
            DC.append(dc)

        if train:
            self.par['Layer' + str(0)] = saab_cur
            self.Energy.append(np.concatenate(eng, axis=0))
        return transformed, DC


    def fit(self, X, diffs, label):
        #        output, DC = [], []
        X, dc = self.cwSaab_1_layer(X, train=True, diffs=diffs, label=label)
        #        output.append(X)
        #        DC.append(dc)
        self.trained = True

    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        output, DC = [], []
        X, dc = self.cwSaab_1_layer(X, train=False)
        output.append(X)
        DC.append(dc)
        return output, DC

    def transform_full_0(self, X, th):

        def Shrink(X, shrinkArg):
            win = shrinkArg['win']
            stride = 1
            new_X = []
            for each_map in X:
                new_X.append(cv2.copyMakeBorder(each_map, win // 2, win // 2, win // 2, win // 2, cv2.BORDER_REFLECT))
            X = np.array(new_X)
            if len(X.shape) == 3:
                X = X[:, :, :, None]
            X = view_as_windows(X, (1, win, win, 1), (1, stride, stride, 1))
            return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

        saab_cur = self.par['Layer' + str(0)]
        transformed, eng, DC = [], [], []
        SaabArg = self.SaabArgs[0]
        X = Shrink(X, self.shrinkArgs[0])
        S = list(X.shape)
        X = X.reshape(-1, S[-1])
        if SaabArg['num_AC_kernels'] != -1:
            S[-1] = SaabArg['num_AC_kernels']
        transformed, dc = saab_cur[0].transform(X)
        transformed = transformed.reshape(S)
        DC.append(dc)

        energy = saab_cur[0].Energy / np.sum(saab_cur[0].Energy)
        total = 0
        idx = transformed.shape[-1] + 1

        for i in range(len(energy)):
            total += energy[i]
            if total > th:
                idx = i + 1
                break
        transformed = transformed[:, :, :, :idx]

        return transformed, DC


if __name__ == "__main__":
    # example useage
    from skimage.util import view_as_windows
    import cv2
    import random
    import matplotlib.pyplot as plt

    random.seed(0)


    # example callback function for collecting patches and its inverse
    def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        stride = shrinkArg['stride']
        new_X = []
        for each_map in X:
            new_X.append(cv2.copyMakeBorder(each_map, win // 2, win // 2, win // 2, win // 2, cv2.BORDER_REFLECT))
        X = np.array(new_X)
        if len(X.shape) == 3:
            X = X[:, :, :, None]
        X = view_as_windows(X, (1, win, win, 1), (1, stride, stride, 1))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)


    # read data
    ori_images, stego_images = readImgToNumpy_pairs('/mnt/zhengwen/image_steganalysis/new_100/ori',
                                                    '/mnt/zhengwen/image_steganalysis/new_100/SUNI_08', mode=0)
    ori_images = ori_images.astype("double")
    stego_images = stego_images.astype("double")
    diffs = stego_images - ori_images
    # set args
    SaabArgs = [{'num_AC_kernels': -1, 'needBias': False, 'useDC': False, 'cw': False},
                {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'cw': True}]
    shrinkArgs = [{'func': Shrink, 'win': 5, 'stride': 1},
                  {'func': Shrink, 'win': 5, 'stride': 1}]

    print(" -----> depth=2")
    p2_unc = PointLabelcwSaab(depth=1, energyTH=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, )
    p2_unc.fit(ori_images, diffs=diffs, label=1)

    output_ori, dc0 = p2_unc.transform_full_0(ori_images, 0.98)
    print(output_ori.shape)
    print("------- DONE -------\n")
    print("Begin to save.")
    np.save('100_potential_1hop.npy', output_ori)

    print("Extracting Context")
    diff_maps = np.squeeze(ori_images - stego_images)

    ori_changed_pts = output_ori[np.where(diff_maps != 0)]

    ori_unchanged_position = np.load('ori_unchanged_position.npy')
    pos_matrix = (ori_unchanged_position[:, 0], ori_unchanged_position[:, 1], ori_unchanged_position[:, 2])
    ori_unchanged_pts = output_ori[pos_matrix]

    label = np.concatenate((1 * np.ones(len(ori_changed_pts)), (-1) * np.ones(len(ori_unchanged_pts))), axis=0)
    train_sample = np.concatenate((ori_changed_pts, ori_unchanged_pts), axis=0)
    idx = np.random.permutation(len(train_sample))
    label = label[idx]
    train_sample = train_sample[idx]

    label = label[:2000000]
    train_sample = train_sample[:2000000]

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(train_sample, label)



    ori_unchanged_pts = np.array(ori_unchanged_pts)
    ori_changed_pts = np.array(ori_changed_pts)
    ori_unchanged_position = np.array(ori_unchanged_position)
    np.save('ori_unchanged_pts.npy', ori_unchanged_pts)
    np.save('ori_changed_pts.npy', ori_changed_pts)
    np.save('ori_unchanged_position.npy', ori_unchanged_position)

    for i in range(ori_changed_pts.shape[-1]):
        plt.figure()
        name = 'dist_ori_img' + str(i) + '.png'
        title = 'dist_ori_img' + str(i)
        plt.hist(ori_changed_pts[:, i], bins=50, color='b', label='changed')
        plt.hist(ori_unchanged_pts[:, i], bins=50, color='r', label='unchanged')
        plt.legend()
        plt.title(title)
        plt.savefig(name)

    ori_changed_pts_abs_sum = np.sum(np.sqrt(np.abs(ori_changed_pts)), axis=1)
    ori_unchanged_pts_abs_sum = np.sum(np.sqrt(np.abs(ori_unchanged_pts)), axis=1)
    plt.figure()
    name = 'dist_ori_pts_abs_sum.png'
    title = 'dist_ori_pts_abs_sum'
    plt.hist(ori_changed_pts_abs_sum, bins=50, color='b', label='changed')
    plt.hist(ori_unchanged_pts_abs_sum, bins=50, color='r', label='unchanged')
    plt.legend()
    plt.title(title)
    plt.savefig(name)

    # # stego
    # output_steg, dc0 = p2_unc.transform_full_0(stego_images, 0.98)
    # np.save('100_steg_1hop.npy', output_steg)
    # steg_unchanged_pts = []
    # steg_changed_pts = []
    # for k in range(len(diff_maps)):
    #     output_map = output_steg[k]
    #     diff_map = diff_maps[k]
    #     width, height, channel = diff_map.shape
    #     for i in range(width):
    #         for j in range(height):
    #             if diff_map[i, j, 0] != 0:
    #                 steg_changed_pts.append(output_map[i, j, :])
    #
    # for each_pos in ori_unchanged_position:
    #     steg_unchanged_pts.append(output_steg[each_pos[0], each_pos[1], each_pos[2], :])
    #
    # steg_unchanged_pts = np.array(steg_unchanged_pts)
    # steg_changed_pts = np.array(steg_changed_pts)
    # np.save('steg_unchanged_pts.npy', steg_unchanged_pts)
    # np.save('steg_changed_pts.npy', steg_changed_pts)
    #
    # for i in range(steg_changed_pts.shape[-1]):
    #     plt.figure()
    #     name = 'dist_steg_img' + str(i) + '.png'
    #     title = 'dist_steg_img' + str(i)
    #     plt.hist(steg_changed_pts[:, i], bins=50, color='b', label='changed')
    #     plt.hist(steg_unchanged_pts[:, i], bins=50, color='r', label='unchanged')
    #     plt.legend()
    #     plt.title(title)
    #     plt.savefig(name)
    #
    # for i in range(steg_changed_pts.shape[-1]):
    #     plt.figure()
    #     name = 'dist_steg_pts' + str(i) + '.png'
    #     title = 'dist_steg_pts' + str(i)
    #     plt.hist(steg_changed_pts[:, i], bins=50, color='b', label='steg')
    #     plt.hist(ori_changed_pts[:, i], bins=50, color='r', label='ori')
    #     plt.legend()
    #     plt.title(title)
    #     plt.savefig(name)
    #
    # for i in range(steg_changed_pts.shape[-1]):
    #     plt.figure()
    #     name = 'dist_bk_pts' + str(i) + '.png'
    #     title = 'dist_bk_pts' + str(i)
    #     plt.hist(steg_unchanged_pts[:, i], bins=50, color='b', label='steg')
    #     plt.hist(ori_unchanged_pts[:, i], bins=50, color='r', label='ori')
    #     plt.legend()
    #     plt.title(title)
    #     plt.savefig(name)
