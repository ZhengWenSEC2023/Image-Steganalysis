# 2020.04.09
import numpy as np
from PCA.cwSaab import cwSaab
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def readImgToNumpy(image_dir):
    """
    Read images in a dir into a numpy array
    """
    img_names = os.listdir(image_dir)
    imgs = []
    for img_name in img_names:
        img = cv2.imread(os.path.join(image_dir, img_name), 0)[:, :, None]
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs


def readImgToNumpy_pairs(ori_dir, stego_dir, mode):
    """
    read paired images sequentially, return ori_imgs and stego_imgs,
    image names based on ori_dir
    mode=0: grayscale
    mode=1: color
    """
    img_names = os.listdir(ori_dir)
    ori_imgs = []
    stego_imgs = []
    for img_name in img_names:
        if mode == 0:
            ori_img = cv2.imread(os.path.join(ori_dir, img_name), mode)[:, :, None]
            stego_img = cv2.imread(os.path.join(stego_dir, img_name), mode)[:, :, None]
        else:
            ori_img = cv2.imread(os.path.join(ori_dir, img_name), mode)
            stego_img = cv2.imread(os.path.join(stego_dir, img_name), mode)
        ori_imgs.append(ori_img)
        stego_imgs.append(stego_img)
    ori_imgs = np.array(ori_imgs)
    stego_imgs = np.array(stego_imgs)
    return ori_imgs, stego_imgs


class Pixelhop2(cwSaab):
    def __init__(self, depth=1, TH1=0.005, TH2=0.001, SaabArgs=None, shrinkArgs=None):
        super().__init__(depth=depth, energyTH=TH1, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs)
        self.TH1 = TH1
        self.TH2 = TH2
        self.idx = []

    def select_(self, X):
        # print('select discarded nodes')
        for i in range(self.depth):
            # print('depth {}: shape before = {}'.format(i,X[i].shape))
            X[i] = X[i][:, :, :, self.Energy[i] >= self.TH2]
            # print('depth {}: shape after = {}'.format(i,X[i].shape))
        return X

    def fit(self, X):
        print('pixelhop2 fit')
        super().fit(X)

    def transform(self, X):
        print('pixelhop2 transform')
        X, _ = super().transform(X)
        X = self.select_(X)
        return X


if __name__ == "__main__":
    # example usage
    from skimage.util import view_as_windows
    import cv2
    import os
    import random

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
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1, X.shape[3])


    # read data
    ori_images, stego_images = readImgToNumpy_pairs('/mnt/zhengwen/image_steganalysis/new_100/ori',
                                                    '/mnt/zhengwen/image_steganalysis/new_100/SUNI_08', mode=0)
    ori_images = ori_images.astype("double")
    stego_images = stego_images.astype("double")

    # set args
    SaabArgs = [{'num_AC_kernels': -1, 'needBias': False, 'useDC': False, 'cw': False},
                {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'cw': True}]
    shrinkArgs = [{'func': Shrink, 'win': 5, 'stride': 2},
                  {'func': Shrink, 'win': 5, 'stride': 2}]

    print(" -----> depth=2")
    p2 = Pixelhop2(depth=1, TH1=0.005, TH2=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs)
    p2.fit(ori_images)

    output_ori, dc0 = p2.transform_full_0(ori_images, 0.98)
    print(output_ori.shape)
    print("------- DONE -------\n")
    print("Begin to save.")
    np.save('100_ori_1hop.npy', output_ori)

    print("Extracting Context")
    diff_maps = ori_images - stego_images
    ori_unchanged_pts = []
    ori_changed_pts = []
    ori_unchanged_position = []
    for k in range(len(diff_maps)):
        output_map = output_ori[k]
        diff_map = diff_maps[k]
        width, height, channel = diff_map.shape
        num_diff = np.sum(diff_map != 0)
        total_num = width * height
        num_same = total_num - num_diff
        for i in range(width):
            for j in range(height):
                if diff_map[i, j, 0] != 0:
                    ori_changed_pts.append(output_map[i, j, :])
                elif random.random() < (num_diff / num_same):
                    ori_unchanged_position.append((k, i, j))
                    ori_unchanged_pts.append(output_map[i, j, :])

    ori_unchanged_pts = np.array(ori_unchanged_pts)
    ori_changed_pts = np.array(ori_changed_pts)
    ori_unchanged_position = np.array(ori_unchanged_position)
    np.save('ori_unchanged_pts.npy', ori_unchanged_pts)
    np.save('ori_changed_pts.npy', ori_changed_pts)
    np.save('ori_unchanged_position.npy', ori_unchanged_position)

    label = np.concatenate((1 * np.ones(len(ori_changed_pts)), (-1) * np.ones(len(ori_unchanged_pts))), axis=0)
    train_sample = np.concatenate((ori_changed_pts, ori_unchanged_pts), axis=0)
    idx = np.random.permutation(len(train_sample))
    label = label[idx]
    train_sample = train_sample[idx]

    label = label[:2000000]
    train_sample = train_sample[:2000000]

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(train_sample, label)

    import pickle

    f = open('oriPH_RMclf_2M.pkl', 'wb')
    pickle.dump(clf, f)
    f.close()

    ori_test, stego_test = readImgToNumpy_pairs('/mnt/zhengwen/image_steganalysis/new_100/test/ori',
                                                '/mnt/zhengwen/image_steganalysis/new_100/test/SUNI_08', mode=0)
    output_test, dc0 = p2.transform_full_0(ori_test, 0.98)
    diff_maps_test = ori_test - stego_test
    ori_unchanged_pts_test = []
    ori_changed_pts_test = []
    ori_unchanged_position_test = []
    for k in range(len(diff_maps_test)):
        output_map = output_test[k]
        diff_map = diff_maps_test[k]
        width, height, channel = diff_map.shape
        num_diff = np.sum(diff_map != 0)
        total_num = width * height
        num_same = total_num - num_diff
        for i in range(width):
            for j in range(height):
                if diff_map[i, j, 0] != 0:
                    ori_changed_pts_test.append(output_map[i, j, :])
                elif random.random() < (num_diff / num_same):
                    ori_unchanged_position_test.append((k, i, j))
                    ori_unchanged_pts_test.append(output_map[i, j, :])

    ori_unchanged_pts_test = np.array(ori_unchanged_pts_test)
    ori_changed_pts_test = np.array(ori_changed_pts_test)
    ori_unchanged_position_test = np.array(ori_unchanged_position_test)

    np.save('ori_unchanged_pts_test.npy', ori_unchanged_pts_test)
    np.save('ori_changed_pts_test.npy', ori_changed_pts_test)
    np.save('ori_unchanged_position_test.npy', ori_unchanged_position_test)

    label_test = np.concatenate((1 * np.ones(len(ori_changed_pts_test)), (-1) * np.ones(len(ori_unchanged_pts_test))),
                                axis=0)
    train_sample_test = np.concatenate((ori_changed_pts_test, ori_unchanged_pts_test), axis=0)

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

    # stego
    output_steg, dc0 = p2.transform_full_0(stego_images, 0.98)
    np.save('100_steg_1hop.npy', output_steg)
    steg_unchanged_pts = []
    steg_changed_pts = []
    for k in range(len(diff_maps)):
        output_map = output_steg[k]
        diff_map = diff_maps[k]
        width, height, channel = diff_map.shape
        for i in range(width):
            for j in range(height):
                if diff_map[i, j, 0] != 0:
                    steg_changed_pts.append(output_map[i, j, :])

    for each_pos in ori_unchanged_position:
        steg_unchanged_pts.append(output_steg[each_pos[0], each_pos[1], each_pos[2], :])

    steg_unchanged_pts = np.array(steg_unchanged_pts)
    steg_changed_pts = np.array(steg_changed_pts)
    np.save('steg_unchanged_pts.npy', steg_unchanged_pts)
    np.save('steg_changed_pts.npy', steg_changed_pts)

    for i in range(steg_changed_pts.shape[-1]):
        plt.figure()
        name = 'dist_steg_img' + str(i) + '.png'
        title = 'dist_steg_img' + str(i)
        plt.hist(steg_changed_pts[:, i], bins=50, color='b', label='changed')
        plt.hist(steg_unchanged_pts[:, i], bins=50, color='r', label='unchanged')
        plt.legend()
        plt.title(title)
        plt.savefig(name)

    for i in range(steg_changed_pts.shape[-1]):
        plt.figure()
        name = 'dist_steg_pts' + str(i) + '.png'
        title = 'dist_steg_pts' + str(i)
        plt.hist(steg_changed_pts[:, i], bins=50, color='b', label='steg')
        plt.hist(ori_changed_pts[:, i], bins=50, color='r', label='ori')
        plt.legend()
        plt.title(title)
        plt.savefig(name)

    for i in range(steg_changed_pts.shape[-1]):
        plt.figure()
        name = 'dist_bk_pts' + str(i) + '.png'
        title = 'dist_bk_pts' + str(i)
        plt.hist(steg_unchanged_pts[:, i], bins=50, color='b', label='steg')
        plt.hist(ori_unchanged_pts[:, i], bins=50, color='r', label='ori')
        plt.legend()
        plt.title(title)
        plt.savefig(name)
