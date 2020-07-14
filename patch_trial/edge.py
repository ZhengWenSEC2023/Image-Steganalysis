# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 00:27:01 2020

@author: Lenovo
"""

import cv2
import numpy as np
import sklearn
import os
import skimage
from sklearn.decomposition import PCA


def sobel(gray_image):
    def gradNorm(grad):
        return (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
    
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 1, 1, cv2.BORDER_DEFAULT)
    m, n = gray_image.shape[0], gray_image.shape[1]
    panel_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    panel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # panel_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    # panel_Y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
    X = cv2.filter2D(gray_image, -1, panel_X)
    Y = cv2.filter2D(gray_image, -1, panel_Y)
    grad = np.sqrt((X ** 2) + (Y ** 2))
    return gradNorm(grad).astype("double")


original_path = r'G:\new_image_stego\BOSSbase_resize'
original_sobel_path = r'G:\new_image_stego\BOSSbase_resize_edges'
stego_path = r'G:\new_image_stego\BOSSbase_S_UNIWARD_05'
stego_sobel_path = r'G:\new_image_stego\BOSSbase_S_UNIWARD_05_edges'

image_names = os.listdir(original_path)
image_names.sort(key=lambda x:int(x[:-4]))

original_vectors, stego_vectors = [], []

count = 0
for image_name in image_names:
    print(count)
    ori_image = cv2.imread( os.path.join(original_path, image_name), 0 )
    stego_image = cv2.imread( os.path.join(stego_path, image_name), 0 )
    
    ori_grad = sobel(ori_image)
    stego_grad = sobel(stego_image)
    
    np.save( os.path.join(original_sobel_path, image_name), (ori_grad / np.sum(ori_grad)))
    np.save( os.path.join(stego_sobel_path, image_name), stego_grad / np.sum(stego_grad))
    count += 1
        
    # ori_block = skimage.util.view_as_blocks(ori_image, (32, 32))
    # stego_block = skimage.util.view_as_blocks(stego_image, (32, 32))
    
    # original_vectors.append(ori_block.reshape((-1, 32 * 32)))
    # stego_vectors.append(stego_block.reshape((-1, 32 * 32)))
    # count += 1
    # if count == 1000:
    #     break

# original_vectors = np.concatenate(original_vectors, axis=0)
# stego_vectors = np.concatenate(stego_vectors, axis=0)
# pca_original, pca_stego = PCA(), PCA()

# pca_original.fit(original_vectors)
# pca_stego.fit(stego_vectors)

# original_components = pca_original.components_
# stego_components = pca_stego.components_

# original_sum_ker = np.sum(original_components, axis=0)
# stego_sum_ker = np.sum(stego_components, axis=0)
# diff_sum_ker = original_sum_ker - stego_sum_ker

# original_abssum_ker = np.sum(abs(original_components), axis=0)
# stego_abssum_ker = np.sum(abs(stego_components), axis=0)
# diff_abssum_ker = original_abssum_ker - stego_abssum_ker

# diff_components = original_components - stego_components

# coeff = original_components * stego_components

# op1=np.sqrt(np.sum(np.square(original_components - stego_components)))
