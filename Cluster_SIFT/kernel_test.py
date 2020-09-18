import cv2
import matplotlib.pyplot as plt
import numpy as np
from Cluster_SIFT.pixelhop2 import Pixelhop2
from skimage.util import view_as_windows

single_circle = cv2.imread(r"C:\Users\Lenovo\Desktop\square.png", 0)
single_circle = cv2.resize(single_circle, (160, 160))
ret, single_circle = cv2.threshold(single_circle, 200, 255, cv2.THRESH_BINARY)

single_size = 8
multip_circle = np.zeros((160, 160))
single_small_circle = cv2.resize(single_circle, (single_size, single_size))
for i in range(160 // single_size):
    for j in range(160 // single_size):
        multip_circle[single_size * i: single_size * (i + 1), 
                      single_size * j: single_size * (j + 1)] = single_small_circle.copy()

single_circle = (single_circle / 255).astype('double')
multip_circle = (multip_circle / 255).astype('double')
# plt.figure()
# plt.imshow(single_circle)
# plt.figure()
# plt.imshow(multip_circle)

def Shrink(X, shrinkArg):
    win = shrinkArg['win']
    X = view_as_windows(X, (1, win, win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)


def Concat(X, concatArg):
    return X

SaabArgs = [{'num_AC_kernels': -1, 'needBias': True, 'useDC': False, 'batch': None},
            {'num_AC_kernels': -1, 'needBias': True, 'useDC': False, 'batch': None}]
shrinkArgsTrain = [{'func': Shrink, 'win': 5},
                   {'func': Shrink, 'win': 5}]
concatArg = {'func': Concat}

p_single = Pixelhop2(depth=1, TH1=1e-20, TH2=1e-30, SaabArgs=SaabArgs, shrinkArgs=shrinkArgsTrain,
                     concatArg=concatArg).fit(single_circle[None, :, :, None].astype('double'))
p_multip = Pixelhop2(depth=1, TH1=1e-20, TH2=1e-30, SaabArgs=SaabArgs, shrinkArgs=shrinkArgsTrain,
                     concatArg=concatArg).fit(multip_circle[None, :, :, None].astype('double'))

channel_idx = 20

single_BY_single_feature = np.squeeze(p_single.transform(single_circle[None, :, :, None])[0])[:, :, channel_idx]
multip_BY_single_feature = np.squeeze(p_single.transform(multip_circle[None, :, :, None])[0])[:, :, channel_idx]
single_BY_multip_feature = np.squeeze(p_multip.transform(single_circle[None, :, :, None])[0])[:, :, channel_idx]
multip_BY_multip_feature = np.squeeze(p_multip.transform(multip_circle[None, :, :, None])[0])[:, :, channel_idx]

kernel_single_1 = p_single.par["Layer0"][0].Kernels[channel_idx].reshape(5,5)
kernel_multip_1 = p_multip.par["Layer0"][0].Kernels[channel_idx].reshape(5,5)

kernel_single_1_fft = np.fft.fft2(kernel_single_1)
kernel_single_1_fft = np.fft.fftshift(kernel_single_1_fft)
kernel_single_1_fft_amp = np.log(np.abs(kernel_single_1_fft))
kernel_multip_1_fft = np.fft.fft2(kernel_multip_1)
kernel_multip_1_fft = np.fft.fftshift(kernel_multip_1_fft)
kernel_multip_1_fft_amp = np.log(np.abs(kernel_multip_1_fft))

total = np.array([
    single_BY_single_feature, multip_BY_single_feature, single_BY_multip_feature, multip_BY_multip_feature
    ])
total = (total - np.min(total)) / (np.max(total) - np.min(total))

single_BY_single_feature = total[0]
multip_BY_single_feature = total[1]
single_BY_multip_feature = total[2]
multip_BY_multip_feature = total[3]

plt.figure()
plt.title("kernel_single_" + str(channel_idx) + "_fft_amp")
plt.imshow(kernel_single_1_fft_amp, 'gray')
plt.figure()
plt.title("kernel_multip_" + str(channel_idx) + "_fft_amp")
plt.imshow(kernel_multip_1_fft_amp, 'gray')

plt.figure()
plt.title("single_BY_single_" + str(channel_idx) + "_feature")
plt.imshow(single_BY_single_feature, 'gray')
plt.figure()
plt.title("multip_BY_single_" + str(channel_idx) + "_feature")
plt.imshow(multip_BY_single_feature, 'gray')
plt.figure()
plt.title("single_BY_multip_" + str(channel_idx) + "_feature")
plt.imshow(single_BY_multip_feature, 'gray')
plt.figure()
plt.title("multip_BY_multip_" + str(channel_idx) + "_feature")
plt.imshow(multip_BY_multip_feature, 'gray')

max_resopnse = [
    np.max(single_BY_single_feature),
    np.max(single_BY_multip_feature),
    np.max(multip_BY_single_feature),
    np.max(multip_BY_multip_feature),
    ]

min_resopnse = [
    np.min(single_BY_single_feature),
    np.min(single_BY_multip_feature),
    np.min(multip_BY_single_feature),
    np.min(multip_BY_multip_feature),
    ]