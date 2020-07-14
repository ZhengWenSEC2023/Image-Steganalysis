# Alex
# yifanwang0916@outlook.com
# 2019.09.25

# Saab transformation for PixelHop unit
# modeiled from https://github.com/davidsonic/Interpretable_CNN

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from numpy import linalg as LA
from skimage.measure import block_reduce
import pickle
import time
import matplotlib.pyplot as plt


class Saab():
    def __init__(self, kernel_sizes, pca_name=None, num_kernels=None, energy_percent=None, useDC=False, getcov=0):
        self.pca_name = pca_name
        self.kernel_sizes = kernel_sizes
        self.num_kernels = num_kernels
        self.useDC = useDC
        self.energy_percent = energy_percent
        self.getcov = getcov
        # self.kernels = []
        # self.bias = []
        self.energy = []
        self.pca_params = []

    def remove_mean(self, features, axis):
        feature_mean = np.mean(features, axis=axis, keepdims=True)
        feature_remove_mean = features - feature_mean
        return feature_remove_mean, feature_mean

    def find_kernels_pca(self, samples, num_kernels, energy_percent=None):
        print("       <PCA Info>        Learning PCA kernel...")
        # pca = IncrementalPCA(n_components=samples.shape[1],batch_size=10000)
        pca = PCA(n_components=samples.shape[1], svd_solver='full')
        pca.fit(samples)
        if self.getcov == 1:
            cov = pca.get_covariance()
        else:
            cov = []
        energy = np.cumsum(pca.explained_variance_ratio_)

        if not energy_percent is None:

            num_components = np.sum(energy <= energy_percent)
            self.num_kernels = num_components
        else:
            num_components = num_kernels
        kernels = pca.components_[:num_components, :]
        mean = pca.mean_
        print("       <PCA Info>        Num of kernels: %d" % num_components)
        print("       <PCA Info>        Energy percent: %f" % np.cumsum(pca.explained_variance_ratio_)[
            num_components - 1])
        ac_energy_normalized = 1 - np.cumsum(pca.explained_variance_ratio_)[:num_components]
        ac_energy = pca.explained_variance_[:num_components]
        return kernels, mean, cov, ac_energy

    def find_kernels_pca_frequency_5(self, samples, num_kernels, high_freq_precentage=None):
        print("       <PCA Info>        Learning PCA kernel...")
        # pca = IncrementalPCA(n_components=samples.shape[1],batch_size=10000)
        pca = PCA(n_components=samples.shape[1], svd_solver='full')
        pca.fit(samples)
        if self.getcov == 1:
            cov = pca.get_covariance()
        else:
            cov = []

        kernels_low_freq = np.array(
            [[0, 0, 0, 0, 0, ],
             [0, 0, 1, 0, 0, ],
             [0, 1, 1, 1, 0, ],
             [0, 0, 1, 0, 0, ],
             [0, 0, 0, 0, 0, ], ]
        )
        selected_idx = []
        if high_freq_precentage is not None:
            for i in range(len(pca.components_)):
                each_kernel = pca.components_[i]
                each_kernel = np.reshape(each_kernel, (5, 5))
                each_fft = abs(np.fft.fftshift(np.fft.fft2(each_kernel)))
                low_freq_engy = np.sum(each_fft * kernels_low_freq)
                total_engy = np.sum(each_fft)
                each_high_percentage = 1 - (low_freq_engy / total_engy)
                if each_high_percentage > high_freq_precentage:
                    selected_idx.append(i)
        else:
            selected_idx = np.array(range(25))
        kernels = pca.components_[selected_idx, :]
        mean = pca.mean_
        self.num_kernels = len(selected_idx)
        print("       <PCA Info>        Num of kernels: %d" % len(selected_idx))
        ac_energy = pca.explained_variance_[selected_idx]
        print("       <PCA Info>        Selected Energy %: ", np.sum(ac_energy) / np.sum(pca.explained_variance_) * 100)
        return kernels, mean, cov, ac_energy

    def remove_zero_patch(self, samples):
        std_var = (np.std(samples, axis=1)).reshape(-1, 1)
        ind_bool = (std_var < 1e-5)
        ind = np.where(ind_bool == True)[0]
        print('zero patch shape:', ind.shape)
        samples_new = np.delete(samples, ind, 0)
        return samples_new

    def Saab_transform(self, feature, kernel_sizes, num_kernels, energy_percent, useDC):
        S = feature.shape
        print("       <PCA Info>        feature.shape: %s" % str(feature.shape))
        if len(feature.shape) > 2:
            sample_patches = feature.reshape(S[0] * S[1] * S[2], -1)
        else:
            sample_patches = np.copy(feature)
        pca_params = {}
        pca_params['kernel_size'] = kernel_sizes
        sample_patches_centered, feature_expectation = self.remove_mean(sample_patches, axis=0)
        training_data, dc = self.remove_mean(sample_patches_centered, axis=1)
        # kernels, mean, cov, ac_energy = self.find_kernels_pca_frequency_5(training_data, num_kernels, high_freq_precentage=energy_percent)
        kernels, mean, cov, ac_energy = self.find_kernels_pca(training_data, num_kernels, energy_percent=energy_percent)
        num_channels = sample_patches.shape[1]
        if useDC == True:
            largest_ev = np.var(dc * np.sqrt(num_channels))
            dc_kernel = 1 / np.sqrt(num_channels) * np.ones((1, num_channels)) / np.sqrt(largest_ev)
            #            dc_kernel = 1 / np.sqrt(num_channels) * np.ones((1, num_channels))
            kernels = np.concatenate((dc_kernel, kernels), axis=0)
            energy = np.concatenate((np.array([largest_ev]), ac_energy), axis=0)
        else:
            energy = ac_energy
        energy = energy / np.sum(energy)  # normalize energy
        transformed = np.matmul(sample_patches_centered, np.transpose(kernels))

        bias = LA.norm(transformed, axis=1)
        bias = np.max(bias)
        # pca_params['Layer_%d/bias' % 0] = bias
        # self.bias = bias
        #        print("       <Info>        Sample patches shape after flatten: %s"%str(sample_patches.shape))
        print("       < PCA Info>        (num of kernel, each kernel shape): ",
              kernels.reshape((-1, int(np.sqrt(kernels.shape[-1] / num_channels)),
                               int(np.sqrt(kernels.shape[-1] / num_channels)), num_channels)).shape)
        print("       < PCA Info>        Transformed shape: %s" % str(transformed.shape))
        pca_params['feature_expectation'] = feature_expectation
        pca_params['kernel'] = kernels
        pca_params['bias'] = bias
        pca_params['pca_mean'] = mean
        pca_params['cov'] = cov
        pca_params['ac_energy'] = ac_energy
        self.pca_params = pca_params
        self.energy = energy
        return transformed

    def fit(self, feature):
        print("------------------- Start: Saab transformation")
        t0 = time.time()
        _ = self.Saab_transform(feature=feature,
                                kernel_sizes=self.kernel_sizes,
                                num_kernels=self.num_kernels,
                                energy_percent=self.energy_percent,
                                useDC=self.useDC)
        if not self.pca_name is None:
            fw = open(self.pca_name, 'wb')
            pickle.dump(self.pca_params, fw)
            fw.close()
            print("       < PCA Info>        Save pca params as name: %s" % str(self.pca_name))
        print("------------------- End: Saab transformation -> using %10f seconds" % (time.time() - t0))
        # return pca_params

    def fit_transform(self, feature):
        print("------------------- Start: Saab transformation")
        t0 = time.time()
        transformed = self.Saab_transform(feature=feature,
                                          kernel_sizes=self.kernel_sizes,
                                          num_kernels=self.num_kernels,
                                          energy_percent=self.energy_percent,
                                          useDC=self.useDC)
        if not self.pca_name is None:
            fw = open(self.pca_name, 'wb')
            pickle.dump(self.pca_params, fw)
            fw.close()
            print("       <Info>        Save pca params as name: %s" % str(self.pca_name))
        print("------------------- End: Saab transformation -> using %10f seconds" % (time.time() - t0))
        # return pca_params
        return transformed
