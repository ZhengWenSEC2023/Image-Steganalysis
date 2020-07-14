

import numpy as np

def calculate_cross_entropy(data, label, class_frequency, bins=10 ):
    # Function to calculate cross entropy of feature dimensions

    # Input:
    # data - output of LAG (nunber of training samples, number of dimensions)
    # label - class labels (nunber of training samples, 1)
    # bins - Number of bins for histogram (default - 10 bins)
    # class_frequency - array containing number of samples for each class (number of classes, 1)

    # Output:
    # cross_entropy - normalized cross_entropy of every dimension in the data (number of dimesnsions, )

    n_classes = np.unique(label).shape[0]
    mini = data.min(axis = 0).reshape((1,data.shape[1]))
    maxi = data.max(axis = 0).reshape((1,data.shape[1]))
    data = (data-mini)/(maxi-mini)

    binwise_frequency = np.zeros((bins,data.shape[1],n_classes))    # 10, 30, 2

    for i in range(data.shape[1]):     # number of features
        for j in range(data.shape[0]):   # number of samples
            desired_bin = np.ceil(data[j][i] * bins) - 1
            binwise_frequency[int(desired_bin)][i][label[j]] += 1

    binwise_frequency = binwise_frequency / class_frequency
    print(binwise_frequency.shape)

    binwise_class = np.argmax(binwise_frequency, axis=2)
    print("bin wise class shape:", binwise_class)

    correct_class_probability = np.zeros((data.shape[1],data.shape[0]))   #p(n,c) 30, 20000
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            correct_class_probability[i][j] = np.count_nonzero(binwise_class.T[i] == label[j]) / bins

    log = -1*np.log(correct_class_probability+1e-9)
    cross_entropy = np.sum(log,axis = 1)
    mini = np.amin(cross_entropy)
    maxi = np.amax(cross_entropy)
    cross_entropy = (cross_entropy - mini) / (maxi - mini)

    return cross_entropy


