import os
import pickle
import numpy as np
import cv2
from Cluster_SIFT.pixelhop2 import Pixelhop2
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.util.shape import view_as_blocks


def diff_sample(diff_map):
    diff_map = diff_map.copy()
    win = 5
    visit_map = np.zeros(diff_map.shape)
    for i in range(diff_map.shape[0]):
        for j in range(diff_map.shape[1]):
            if visit_map[i, j] == 0:
                if diff_map[i, j] != 0:
                    diff_map[max(0, i - win): min(i + win + 1, 256), max(0, j - win): min(j + win + 1, 256)] = 0
                    diff_map[i, j] = 1
                    visit_map[max(0, i - win): min(i + win + 1, 256), max(0, j - win): min(j + win + 1, 256)] = 1
    return diff_map


P_TRAIN_NUM_TOTAL = 1500
BLOCK_SIZE = 32
KERNEL_NUM = 6

print(P_TRAIN_NUM_TOTAL)
np.random.seed(23)

ori_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_resize'
steg_img_path = r'/mnt/zhengwen/new_trial/BOSSbase_S_UNIWARD_05'
image_shape = [256, 256]
keypoint = [cv2.KeyPoint(x + BLOCK_SIZE // 2, y + BLOCK_SIZE // 2, BLOCK_SIZE)
            for y in range(0, image_shape[0], BLOCK_SIZE)
            for x in range(0, image_shape[1], BLOCK_SIZE)]
sift = cv2.xfeatures2d.SIFT_create()

blocks_stego_train = []
blocks_cover_train = []
blocks_diff = []
blocks_dense_feat = []
count = 0
for file_name in sorted(os.listdir(ori_img_path), key=lambda x: int(x[:-4])):
    # print(count)
    ori_img = cv2.imread(os.path.join(ori_img_path, file_name), 0)[:, :, None]
    steg_img = cv2.imread(os.path.join(steg_img_path, file_name), 0)[:, :, None]
    diff_map = np.squeeze(ori_img.astype("double") - steg_img.astype("double"))
    ori_block = view_as_blocks(ori_img, block_shape=(BLOCK_SIZE, BLOCK_SIZE, 1)).reshape(-1, BLOCK_SIZE, BLOCK_SIZE, 1)
    steg_block = view_as_blocks(steg_img, block_shape=(BLOCK_SIZE, BLOCK_SIZE, 1)).reshape(-1, BLOCK_SIZE, BLOCK_SIZE, 1)
    diff_block = view_as_blocks(diff_map, block_shape=(BLOCK_SIZE, BLOCK_SIZE)).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
    blocks_dense_feat.append(sift.compute(np.squeeze(ori_img), keypoint)[1])
    blocks_stego_train.append(steg_block)
    blocks_cover_train.append(ori_block)
    blocks_diff.append(abs(diff_block))
    count += 1
    if count == P_TRAIN_NUM_TOTAL:
        break

blocks_cover_train = np.concatenate(blocks_cover_train, axis=0).astype("double")
blocks_stego_train = np.concatenate(blocks_stego_train, axis=0).astype("double")
blocks_diff = np.concatenate(blocks_diff, axis=0).astype("double")
blocks_dense_feat = np.concatenate(blocks_dense_feat, axis=0).astype("double")

kmeans = KMeans(n_clusters=KERNEL_NUM).fit(blocks_dense_feat)
labels = kmeans.labels_

count_for_total = []
print("% of each CLUSTER: ")
for i in range(KERNEL_NUM):
    count_for_total.append(np.sum(kmeans.labels_ == i) / len(kmeans.labels_))
    print(i, count_for_total[i])

diff_rate = []
print("AVG % of stego pixel: ")
for i in range(KERNEL_NUM):
    idx = kmeans.labels_ == i
    cur_cluster = blocks_diff[idx]
    diff_rate.append(np.sum(cur_cluster) / (cur_cluster.shape[0] * BLOCK_SIZE * BLOCK_SIZE))
    print(i, diff_rate[i])

steg_top_class = np.argmax(diff_rate)
steg_low_class = np.argmin(diff_rate)

print("MAX steg class:", "COUNT for:", count_for_total[steg_top_class], "% STEG", diff_rate[steg_top_class])
print("MIN steg class:", "COUNT for:", count_for_total[steg_low_class], "% STEG", diff_rate[steg_low_class])


# PixlHop ++
def Shrink(X, shrinkArg, max_pooling=True, padding=True):
    X = view_as_windows(X, (1, 5, 5, 1))
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
    return X


# def ShrinkTest(X, shrinkArg, max_pooling=True, padding=True):
#     X = view_as_windows(X, (1, 5, 5, 1))
#     X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
#     return X


# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X


# set args for PixelHop++
SaabArgs = [{'num_AC_kernels': -1, 'needBias': True, 'useDC': False, 'batch': None},
            {'num_AC_kernels': -1, 'needBias': True, 'useDC': False, 'batch': None}]
shrinkArgsTrain = [{'func': Shrink, 'win': 5},
                   {'func': Shrink, 'win': 5}]
concatArg = {'func': Concat}

p1_max = Pixelhop2(depth=1, TH1=1e-20, TH2=1e-30, SaabArgs=SaabArgs, shrinkArgs=shrinkArgsTrain,
                   concatArg=concatArg).fit(blocks_cover_train[kmeans.labels_ == steg_top_class])
f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PXH_patch_energy_MAX.pkl", 'wb') # 1st hop
pickle.dump(p1_max, f)
f.close()

features_cover_max_p1 = p1_max.transform(blocks_cover_train[kmeans.labels_ == steg_top_class])
features_stego_max_p1 = p1_max.transform(blocks_stego_train[kmeans.labels_ == steg_top_class])

feature_max_p1 = np.concatenate((features_cover_max_p1[0], features_stego_max_p1[0]), axis=0)
feature_max_p1_shape = feature_max_p1.shape
feature_max_p1 = feature_max_p1.reshape(-1, feature_max_p1_shape[-1])
feature_max_p1_scaled = feature_max_p1 - np.mean(feature_max_p1, axis=0)
feature_max_p1_scaled = feature_max_p1_scaled.reshape(feature_max_p1_shape)

features_cover_max_p1_energy = np.sum(np.sum(np.square(feature_max_p1_scaled[:(feature_max_p1_scaled.shape[0] // 2), :, :, :]), axis=1), axis=1) / (feature_max_p1_scaled.shape[1] * feature_max_p1_scaled.shape[2])
features_stego_max_p1_energy = np.sum(np.sum(np.square(feature_max_p1_scaled[(feature_max_p1_scaled.shape[0] // 2):, :, :, :]), axis=1), axis=1) / (feature_max_p1_scaled.shape[1] * feature_max_p1_scaled.shape[2])
np.save('/mnt/zhengwen/image_steganalysis/dataset/codes/features_cover_max_p1_energy.npy', features_cover_max_p1_energy)
np.save('/mnt/zhengwen/image_steganalysis/dataset/codes/features_stego_max_p1_energy.npy', features_stego_max_p1_energy)

for i in range(features_cover_max_p1[0].shape[-1]):
    p2_max = Pixelhop2(depth=1, TH1=1e-20, TH2=1e-30, SaabArgs=SaabArgs, shrinkArgs=shrinkArgsTrain,
                       concatArg=concatArg).fit(features_cover_max_p1[0][:, :, :, i][:, :, :, None])
    f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PXH_patch_energy_MAX_" + str(i) + "_p2.pkl", 'wb') # 2nd hop
    pickle.dump(p2_max, f)
    f.close()

    features_cover_max_p2 = p2_max.transform(features_cover_max_p1[0][:, :, :, i][:, :, :, None])
    features_stego_max_p2 = p2_max.transform(features_stego_max_p1[0][:, :, :, i][:, :, :, None])

    feature_max_p2 = np.concatenate((features_cover_max_p2[0], features_stego_max_p2[0]), axis=0)
    feature_max_p2_shape = feature_max_p2.shape
    feature_max_p2 = feature_max_p2.reshape(-1, feature_max_p2_shape[-1])
    feature_max_p2_scaled = feature_max_p2 - np.mean(feature_max_p2, axis=0)
    feature_max_p2_scaled = feature_max_p2_scaled.reshape(feature_max_p2_shape)

    features_cover_max_p2_energy = np.sum(
        np.sum(np.square(feature_max_p2_scaled[:(feature_max_p2_scaled.shape[0] // 2), :, :, :]), axis=1), axis=1) / (
                                               feature_max_p2_scaled.shape[1] * feature_max_p2_scaled.shape[2])
    features_stego_max_p2_energy = np.sum(
        np.sum(np.square(feature_max_p2_scaled[(feature_max_p2_scaled.shape[0] // 2):, :, :, :]), axis=1), axis=1) / (
                                               feature_max_p2_scaled.shape[1] * feature_max_p2_scaled.shape[2])
    np.save('/mnt/zhengwen/image_steganalysis/dataset/codes/features_cover_max_p2_energy_' + str(i) + '.npy',
            features_cover_max_p2_energy)
    np.save('/mnt/zhengwen/image_steganalysis/dataset/codes/features_stego_max_p2_energy_' + str(i) + '.npy',
            features_stego_max_p2_energy)

print("Begin for min")
p1_min = Pixelhop2(depth=1, TH1=1e-20, TH2=1e-30, SaabArgs=SaabArgs, shrinkArgs=shrinkArgsTrain,
                   concatArg=concatArg).fit(blocks_cover_train[kmeans.labels_ == steg_low_class])
f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PXH_patch_energy_MIN.pkl", 'wb') # 1st hop
pickle.dump(p1_min, f)
f.close()

features_cover_min_p1 = p1_min.transform(blocks_cover_train[kmeans.labels_ == steg_low_class])
features_stego_min_p1 = p1_min.transform(blocks_stego_train[kmeans.labels_ == steg_low_class])

feature_min_p1 = np.concatenate((features_cover_min_p1[0], features_stego_min_p1[0]), axis=0)
feature_min_p1_shape = feature_min_p1.shape
feature_min_p1 = feature_min_p1.reshape(-1, feature_min_p1_shape[-1])
feature_min_p1_scaled = feature_min_p1 - np.mean(feature_min_p1, axis=0)
feature_min_p1_scaled = feature_min_p1_scaled.reshape(feature_min_p1_shape)

features_cover_min_p1_energy = np.sum(np.sum(np.square(feature_min_p1_scaled[:(feature_min_p1_scaled.shape[0] // 2), :, :, :]), axis=1), axis=1) / (feature_min_p1_scaled.shape[1] * feature_min_p1_scaled.shape[2])
features_stego_min_p1_energy = np.sum(np.sum(np.square(feature_min_p1_scaled[(feature_min_p1_scaled.shape[0] // 2):, :, :, :]), axis=1), axis=1) / (feature_min_p1_scaled.shape[1] * feature_min_p1_scaled.shape[2])
np.save('/mnt/zhengwen/image_steganalysis/dataset/codes/features_cover_min_p1_energy.npy', features_cover_min_p1_energy)
np.save('/mnt/zhengwen/image_steganalysis/dataset/codes/features_stego_min_p1_energy.npy', features_stego_min_p1_energy)

for i in range(features_cover_min_p1[0].shape[-1]):
    p2_min = Pixelhop2(depth=1, TH1=1e-20, TH2=1e-30, SaabArgs=SaabArgs, shrinkArgs=shrinkArgsTrain,
                       concatArg=concatArg).fit(features_cover_min_p1[0][:, :, :, i][:, :, :, None])
    f = open("/mnt/zhengwen/image_steganalysis/dataset/codes/PXH_patch_energy_MIN_" + str(i) + "_p2.pkl", 'wb') # 2nd hop
    pickle.dump(p2_min, f)
    f.close()

    features_cover_min_p2 = p2_min.transform(features_cover_min_p1[0][:, :, :, i][:, :, :, None])
    features_stego_min_p2 = p2_min.transform(features_stego_min_p1[0][:, :, :, i][:, :, :, None])

    feature_min_p2 = np.concatenate((features_cover_min_p2[0], features_stego_min_p2[0]), axis=0)
    feature_min_p2_shape = feature_min_p2.shape
    feature_min_p2 = feature_min_p2.reshape(-1, feature_min_p2_shape[-1])
    feature_min_p2_scaled = feature_min_p2 - np.mean(feature_min_p2, axis=0)
    feature_min_p2_scaled = feature_min_p2_scaled.reshape(feature_min_p2_shape)

    features_cover_min_p2_energy = np.sum(
        np.sum(np.square(feature_min_p2_scaled[:(feature_min_p2_scaled.shape[0] // 2), :, :, :]), axis=1), axis=1) / (
                                               feature_min_p2_scaled.shape[1] * feature_min_p2_scaled.shape[2])
    features_stego_min_p2_energy = np.sum(
        np.sum(np.square(feature_min_p2_scaled[(feature_min_p2_scaled.shape[0] // 2):, :, :, :]), axis=1), axis=1) / (
                                               feature_min_p2_scaled.shape[1] * feature_min_p2_scaled.shape[2])
    np.save('/mnt/zhengwen/image_steganalysis/dataset/codes/features_cover_min_p2_energy_' + str(i) + '.npy',
            features_cover_min_p2_energy)
    np.save('/mnt/zhengwen/image_steganalysis/dataset/codes/features_stego_min_p2_energy_' + str(i) + '.npy',
            features_stego_min_p2_energy)


# PLOT
features_cover_max_p1_energy = np.load('/mnt/zhengwen/image_steganalysis/dataset/codes/features_cover_max_p1_energy.npy')
features_stego_max_p1_energy = np.load('/mnt/zhengwen/image_steganalysis/dataset/codes/features_stego_max_p1_energy.npy')
plt.figure()
plt.bar(range(0, 24), abs(np.mean(features_cover_max_p1_energy, axis=0) - np.mean(features_stego_max_p1_energy, axis=0))[:-1])
plt.xlabel("index of channels")
plt.ylabel("energy difference")
plt.title("most stego cluster pixelhop 1")
plt.savefig("P1_energy_difference_MAX.png")
# plt.figure()
# plt.plot(np.mean(features_cover_max_p1_energy, axis=0)[:-1], label='cover')
# plt.plot(np.mean(features_stego_max_p1_energy, axis=0)[:-1], label='stego')
# plt.xlabel("index of channels")
# plt.ylabel("energy")
# plt.title("most stego cluster pixelhop 1")
# plt.legend()
# plt.savefig("P1_energy_difference_MAX_curve.png")

features_cover_max_p2_energy = []
features_stego_max_p2_energy = []
for i in range(25):
    features_cover_max_p2_energy.append(np.load('/mnt/zhengwen/image_steganalysis/dataset/codes/features_cover_max_p2_energy_' + str(i) + '.npy'))
    features_stego_max_p2_energy.append(np.load('/mnt/zhengwen/image_steganalysis/dataset/codes/features_stego_max_p2_energy_' + str(i) + '.npy'))
features_cover_max_p2_energy = np.concatenate(features_cover_max_p2_energy, axis=1)
features_stego_max_p2_energy = np.concatenate(features_stego_max_p2_energy, axis=1)
plt.figure()
plt.bar(range(0, 624), abs(np.mean(features_cover_max_p2_energy, axis=0) - np.mean(features_stego_max_p2_energy, axis=0))[:-1])
plt.xlabel("index of channels")
plt.ylabel("energy difference")
plt.title("most stego cluster pixelhop 2")
plt.savefig("P2_energy_difference_MAX.png")
# plt.figure()
# plt.plot(np.mean(features_cover_max_p2_energy, axis=0)[:-1], label='cover')
# plt.plot(np.mean(features_stego_max_p2_energy, axis=0)[:-1], label='stego')
# plt.xlabel("index of channels")
# plt.ylabel("energy")
# plt.title("most stego cluster pixelhop 2")
# plt.legend()
# plt.savefig("P2_energy_difference_MAX_curve.png")

features_cover_min_p1_energy = np.load('/mnt/zhengwen/image_steganalysis/dataset/codes/features_cover_min_p1_energy.npy')
features_stego_min_p1_energy = np.load('/mnt/zhengwen/image_steganalysis/dataset/codes/features_stego_min_p1_energy.npy')
plt.figure()
plt.bar(range(0, 24), abs(np.mean(features_cover_min_p1_energy, axis=0) - np.mean(features_stego_min_p1_energy, axis=0))[:-1])
plt.xlabel("index of channels")
plt.ylabel("energy difference")
plt.title("least stego cluster pixelhop 1")
plt.savefig("P1_energy_difference_MIN.png")
# plt.figure()
# plt.plot(np.mean(features_cover_min_p1_energy, axis=0)[:-1], label='cover')
# plt.plot(np.mean(features_stego_min_p1_energy, axis=0)[:-1], label='stego')
# plt.xlabel("index of channels")
# plt.ylabel("energy")
# plt.title("least stego cluster pixelhop 1")
# plt.legend()
# plt.savefig("P1_energy_difference_MIN_curve.png")

features_cover_min_p2_energy = []
features_stego_min_p2_energy = []
for i in range(25):
    features_cover_min_p2_energy.append(np.load('/mnt/zhengwen/image_steganalysis/dataset/codes/features_cover_min_p2_energy_' + str(i) + '.npy'))
    features_stego_min_p2_energy.append(np.load('/mnt/zhengwen/image_steganalysis/dataset/codes/features_stego_min_p2_energy_' + str(i) + '.npy'))
features_cover_min_p2_energy = np.concatenate(features_cover_min_p2_energy, axis=1)
features_stego_min_p2_energy = np.concatenate(features_stego_min_p2_energy, axis=1)
plt.figure()
plt.bar(range(0, 624), abs(np.mean(features_cover_min_p2_energy, axis=0) - np.mean(features_stego_min_p2_energy, axis=0))[:-1])
plt.xlabel("index of channels")
plt.ylabel("energy difference")
plt.title("least stego cluster pixelhop 2")
plt.savefig("P2_energy_difference_MIN.png")
# plt.figure()
# plt.plot(np.mean(features_cover_min_p2_energy, axis=0)[:-1], label='cover')
# plt.plot(np.mean(features_stego_min_p2_energy, axis=0)[:-1], label='stego')
# plt.xlabel("index of channels")
# plt.ylabel("energy")
# plt.title("least stego cluster pixelhop 2")
# plt.legend()
# plt.savefig("P2_energy_difference_MIN_curve.png")
