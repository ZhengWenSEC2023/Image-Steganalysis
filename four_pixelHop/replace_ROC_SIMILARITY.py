import pickle
import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import os
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns

def euclidDis(first_vec, second_vec):
    return np.linalg.norm(first_vec - second_vec)


def manhattanDistance(first_vec, second_vec):
    return np.linalg.norm(first_vec - second_vec, ord=1)


def chebyshevDistance(first_vec, second_vec):
    return np.linalg.norm(first_vec - second_vec, ord=np.inf)


def cosSimilarity(first_vec, second_vec):
    return np.dot(first_vec, second_vec) / (np.linalg.norm(first_vec) * np.linalg.norm(second_vec))


context = np.load("unshuffled_clustering_context.npy")
context_label = np.load("unshuffled_clustering_label.npy")

context_remove_mean = context - np.mean(context, axis=0)

changed_context = context_remove_mean[context_label == 1]
unchanged_context = context_remove_mean[context_label == 0]

mean_changed_context = np.mean(changed_context, axis=0)

euclid_changed = []
euclid_unchanged = []
for i in range(changed_context.shape[0]):
    print("E_changed", i, "Total", changed_context.shape[0])
    euclid_changed.append(euclidDis(changed_context[i], mean_changed_context))

for i in range(unchanged_context.shape[0]):
    print("E_unchanged", i, "Total", unchanged_context.shape[0])
    euclid_unchanged.append(euclidDis(unchanged_context[i], mean_changed_context))

manhattan_changed = []
manhattan_unchanged = []
for i in range(changed_context.shape[0]):
    print("M_changed", i, "Total", changed_context.shape[0])
    manhattan_changed.append(manhattanDistance(changed_context[i], mean_changed_context))

for i in range(unchanged_context.shape[0]):
    print("M_unchanged", i, "Total", unchanged_context.shape[0])
    manhattan_unchanged.append(manhattanDistance(unchanged_context[i], mean_changed_context))


chebyshev_changed = []
chebyshev_unchanged = []
for i in range(changed_context.shape[0]):
    print("CHE_changed", i, "Total", changed_context.shape[0])
    chebyshev_changed.append(chebyshevDistance(changed_context[i], mean_changed_context))

for i in range(unchanged_context.shape[0]):
    print("CHE_unchanged", i, "Total", unchanged_context.shape[0])
    chebyshev_unchanged.append(chebyshevDistance(unchanged_context[i], mean_changed_context))


cos_changed = []
cos_unchanged = []
for i in range(changed_context.shape[0]):
    print("COS_changed", i, "Total", changed_context.shape[0])
    cos_changed.append(cosSimilarity(changed_context[i], mean_changed_context))

for i in range(unchanged_context.shape[0]):
    print("CHE_unchanged", i, "Total", unchanged_context.shape[0])
    cos_unchanged.append(cosSimilarity(unchanged_context[i], mean_changed_context))


plt.figure()
sns.distplot(euclid_changed, kde=False, bins=20, label="changed")
sns.distplot(euclid_unchanged, kde=False, bins=20, label="unchanged")
plt.xlabel("Distance")
plt.ylabel("Number of vectors")
plt.title("Euclid Distance")
plt.legend()
plt.savefig("Euclid Distance.png")

plt.figure()
sns.distplot(manhattan_changed, kde=False, bins=20, label="changed")
sns.distplot(manhattan_unchanged, kde=False, bins=20, label="unchanged")
plt.xlabel("Distance")
plt.ylabel("Number of vectors")
plt.title("Manhattan Distance")
plt.legend()
plt.savefig("Manhattan Distance.png")

plt.figure()
sns.distplot(chebyshev_changed, kde=False, bins=20, label="changed")
sns.distplot(chebyshev_unchanged, kde=False, bins=20, label="unchanged")
plt.xlabel("Distance")
plt.ylabel("Number of vectors")
plt.title("Chebyshev Distance")
plt.legend()
plt.savefig("Chebyshev Distance.png")

plt.figure()
sns.distplot(cos_changed, kde=False, bins=20, label="changed")
sns.distplot(cos_unchanged, kde=False, bins=20, label="unchanged")
plt.xlabel("Distance")
plt.ylabel("Number of vectors")
plt.title("Cos Similarity")
plt.legend()
plt.savefig("Cos Similarity.png")
