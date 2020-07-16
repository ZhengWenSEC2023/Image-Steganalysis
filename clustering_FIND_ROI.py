import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import os
from sklearn.cluster import OPTICS
from sklearn.metrics import calinski_harabaz_score

NUM_USED_CONTEXT = 500000
# K_MAX = 15
# ITERATIONS = 10

context = np.load("unshuffled_clustering_context.npy")
context_label = np.load("unshuffled_clustering_label.npy")

idx = np.random.permutation(len(context))
context = context[idx][:NUM_USED_CONTEXT]
context_label = context_label[idx][:NUM_USED_CONTEXT]

num_diff = np.sum(context_label != 0)

clustering_evalution = []

clustering = OPTICS().fit(context)
cluster_label = clustering.labels_
temp_k = {
        "stego_points_in_each_cluster": [],
        "stego_points_in_cluster_count_as_total": [],
        "number_of_samples_in_clusters": [],
        "calinski_harabaz_score": [],
        "clusters": []
    }
for cluster_idx in range(len(np.unique(cluster_label))):
    temp_k["stego_points_in_each_cluster"].append(
        np.sum(context_label[cluster_label == cluster_idx] == 1) / np.sum(cluster_label == cluster_idx)
    )
    temp_k["stego_points_in_cluster_count_as_total"].append(
        np.sum(context_label[cluster_label == cluster_idx] == 1) / np.sum(context_label == 1)
    )
    temp_k["number_of_samples_in_clusters"].append(
        np.sum(cluster_label == cluster_idx)
    )
    temp_k["calinski_harabaz_score"].append(
        calinski_harabaz_score(context, cluster_label)
    )
    temp_k["clusters"].append(clustering)

f = open("clustering_result_OPTICS.pkl", 'wb')
pickle.dump(temp_k, f)
f.close()

# for k in range(K_MIN, K_MAX + 1):
#     temp_k = {
#         "k": k,
#         "iter": ITERATIONS,
#         "stego_points_in_each_cluster": [],
#         "stego_points_in_cluster_count_as_total": [],
#         "number_of_samples_in_clusters": [],
#         "calinski_harabaz_score": [],
#         "clusters": []
#     }
#     for iter in range(ITERATIONS):
#         print("K:", k, "ITER:", iter)
#         clustering = OPTICS(n_clusters=k, random_state=5).fit(context)
#         cluster_label = clustering.labels_
#         for cluster_idx in range(k):
#             temp_k["stego_points_in_each_cluster"].append(
#                 np.sum(context_label[cluster_label == k] == 1) / np.sum(cluster_label == k)
#             )
#             temp_k["stego_points_in_cluster_count_as_total"].append(
#                 np.sum(context_label[cluster_label == k] == 1) / np.sum(context_label == 1)
#             )
#             temp_k["number_of_samples_in_clusters"].append(
#                 np.sum(cluster_label == k)
#             )
#             temp_k["calinski_harabaz_score"].append(
#                 calinski_harabaz_score(context, cluster_label)
#             )
#             temp_k["clusters"].append(clustering)
#     clustering_evalution.append(temp_k)
