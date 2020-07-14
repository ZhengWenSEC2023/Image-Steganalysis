
#%%
import numpy as np
import time
import pickle
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from previous_trial.Hierarchical_kmeans import HierKmeans

warnings.filterwarnings('ignore')





if __name__=='__main__':

    save_dir_matFiles = '/mnt/yaozhu/image_splicing_mnt/Output_Mat_Files/regression_v1/channelwisepca/multi_resolution/features/'
    print("----Training----")

    # read in multi-resolution features


    positive1 = np.load(save_dir_matFiles + 'original_resolution/context/pos_1_100.npy')
    negative1 = pickle.load(open(save_dir_matFiles + 'original_resolution/context/neg_1_100_all.pkl', 'rb'))
    # positive2 = np.load(save_dir_matFiles + '1-2_resolution/ssl/pos_1_100.npy')
    # negative2 = pickle.load(open(save_dir_matFiles + '1-2_resolution/ssl/neg_1_100.pkl', 'rb'))
    # positive4 = np.load(save_dir_matFiles + '1-4_resolution/lab/pos_1_100.npy')
    # negative4 = pickle.load(open(save_dir_matFiles + '1-4_resolution/lab/neg_1_100.pkl', 'rb'))
    # positive8 = np.load(save_dir_matFiles + '1-8_resolution/pos_1_100.npy')[:,:,0]
    # negative8 = pickle.load(open(save_dir_matFiles + '1-8_resolution/neg_1_100.pkl', 'rb'))[:,:,0]


    print("original shape:", positive1.shape, negative1.shape)
    # print("1/2 shape:", positive2.shape, negative2.shape)
    # print("1/4 shape:", positive4.shape, negative4.shape)
    # print("1/8 shape:", positive8.shape, negative8.shape)


    X_1 = np.concatenate((positive1, negative1), axis=0)
    # X_2 = np.concatenate((positive2, negative2), axis=0)
    # X_4 = np.concatenate((positive4, negative4), axis=0)
    # X_8 = np.concatenate((positive8, negative8), axis=0)



    # target is different with differnt weights
    pos_target_1 = np.load(save_dir_matFiles + 'original_resolution/clear_test/pos_target.npy')
    neg_target_1 = pickle.load(open(save_dir_matFiles + 'original_resolution/clear_test/neg_target.pkl', 'rb'))
    #
    # pos_target_2 = np.load(save_dir_matFiles + '1-2_resolution/ssl/pos_target.npy')
    # neg_target_2 = pickle.load(open(save_dir_matFiles + '1-2_resolution/ssl/neg_target.pkl', 'rb'))

    # pos_target_4 = np.load(save_dir_matFiles + '1-4_resolution/lab/pos_target.npy')
    # neg_target_4 = pickle.load(open(save_dir_matFiles + '1-4_resolution/lab/neg_target.pkl', 'rb'))
    #
    # pos_target_8 = np.load(save_dir_matFiles + '1-8_resolution/pos_target.npy')
    # neg_target_8 = pickle.load(open(save_dir_matFiles + '1-8_resolution/neg_target.pkl', 'rb'))



    # read test features, same for different weights
    test_positive1 = np.load(save_dir_matFiles + 'original_resolution/clear_test/test_pos_1.npy')
    test_negative1 = np.load(save_dir_matFiles + 'original_resolution/clear_test/test_neg_1.npy')
    # test_positive2 = np.load(save_dir_matFiles + '1-2_resolution/ssl/test_pos_1.npy')
    # test_negative2 = np.load(save_dir_matFiles + '1-2_resolution/ssl/test_neg_1.npy')
    # test_positive2 = pickle.load(open(save_dir_matFiles + '1-2_resolution/ssl/test_pos_1.pkl', 'rb'))
    # test_negative2 = pickle.load(open(save_dir_matFiles + '1-2_resolution/ssl/test_neg_1.pkl', 'rb'))

    # test_positive4 = np.load(save_dir_matFiles + '1-4_resolution/lab/test_pos_1.npy')
    # test_negative4 = np.load(save_dir_matFiles + '1-4_resolution/lab/test_neg_1.npy')
    # test_positive8 = np.load(save_dir_matFiles + '1-8_resolution/test_pos_1_40.npy')[:,:,0]
    # test_negative8 = np.load(save_dir_matFiles + '1-8_resolution/test_neg_1_40.npy')[:,:,0]

    print("original shape:", test_positive1.shape, test_negative1.shape)
    # print("1/2 shape:", test_positive2.shape, test_negative2.shape)
    # print("1/4 shape:", test_positive4.shape, test_negative4.shape)
    # print("1/8 shape:", test_positive8.shape, test_negative8.shape)



    Y_target_1 = np.concatenate((pos_target_1, neg_target_1), axis=0)
    # Y_target_2 = np.concatenate((pos_target_2, neg_target_2), axis=0)
    # Y_target_4 = np.concatenate((pos_target_4, neg_target_4), axis=0)
    # Y_target_8 = np.concatenate((pos_target_8, neg_target_8), axis=0)


    # eliminate very small target sample
    idx_small = np.where(abs(Y_target_1) < 0.05)
    X = np.delete(X_1, idx_small, axis=0)
    Y = np.delete(Y_target_1, idx_small, axis=0)
    print(X.shape, X_1.shape)


    plt.figure()
    bin_edge = np.linspace(-1, 1, 21, endpoint=True)
    n_p, bins_p, patches_p = plt.hist(x=Y[Y>0], bins=bin_edge, color='b',
                                      alpha=0.7, rwidth=0.9, label="spliced pixels")
    n_n, bins_n, patches_n = plt.hist(x=Y[Y<0], bins=bin_edge, color='y',
                                      alpha=0.7, rwidth=0.9, label="authentic pixels")

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('histogram of train predicted Y for 1/1 resolution')
    plt.legend()

    plt.savefig(save_dir_matFiles + "original_resolution/clear_test/train_target_histogram_1-1.png")





    #          ###########       TRAINING       ##############

    # split 80% and 20% training and validation set

    # X_train_1, X_val_1, Y_train_1, Y_val_1 = train_test_split(X_1, Y_target_1, test_size=0.1, random_state=42)
    X_train_2, X_val_2, Y_train_2, Y_val_2 = train_test_split(X, Y, test_size=0.1, random_state=42)
    # X_train_4, X_val_4, Y_train_4, Y_val_4 = train_test_split(X_4, Y_target_4, test_size=0.1, random_state=42)
    # X_train_8, X_val_8, Y_train_8, Y_val_8 = train_test_split(X_8, Y_target_8, test_size=0.1, random_state=42)


    # Hierarchical Kmeans

    metric = {'min_num_sample': 100,
              'purity': 0.9}
    clf = HierKmeans(depth=25, learner=LinearRegression(), num_cluster=2, metric=metric)
    clf.fit(X_train_2, Y_train_2)
    print(clf.nodes.keys())

    t1 = time.time()
    Y_train_pred = clf.predict_proba(X_train_2)
    t2 = time.time()

    Y_val_pred = clf.predict_proba(X_val_2)

    print("Y_train_pred shape:", Y_train_pred.shape, "Y_val_pred shape:", Y_val_pred.shape)
    print("time for training prediction:", t2-t1)


    # # normalize pred_Y
    # Y_train_pred_mean = np.mean(Y_train_pred)
    # Y_val_pred_mean = np.mean(Y_val_pred)
    # Y_train_pred = Y_train_pred - Y_train_pred_mean
    # Y_val_pred = Y_val_pred - Y_val_pred_mean


    Y_train_label = np.zeros((Y_train_2.shape[0]))
    Y_train_pred_label = np.zeros((Y_train_2.shape[0]))
    Y_train_label[Y_train_2 > 0] = 1
    Y_train_pred_label[Y_train_pred > 0] = 1


    Y_val_label = np.zeros((Y_val_2.shape[0]))
    Y_val_pred_label = np.zeros((Y_val_2.shape[0]))
    Y_val_label[Y_val_2 > 0] = 1
    Y_val_pred_label[Y_val_pred > 0] = 1


    plt.figure()
    n_p, bins_p, patches_p = plt.hist(x=Y_train_pred[Y_train_2>0], bins='auto', color='b', rwidth=0.9,
                                    label="spliced pixels")
    n_n, bins_n, patches_n = plt.hist(x=Y_train_pred[Y_train_2<0], bins='auto', color='y', rwidth=0.9,
                                    label="authentic pixels")

    plt.grid(axis='y')
    plt.xlabel('Predicted Y')
    plt.ylabel('Frequency')
    plt.title('histogram of Training predicted Y')
    plt.legend()

    plt.savefig(save_dir_matFiles + "original_resolution/clear_test/histogram_1-1_lab_HierKM_train_weight_rf.png")


    C_train = metrics.confusion_matrix(Y_train_label, Y_train_pred_label, labels=[0, 1])
    per_class_accuracy_train = np.diag(C_train.astype(np.float32)) / np.sum(C_train.astype(np.float32), axis=1)
    print("train:", per_class_accuracy_train)


    C_val = metrics.confusion_matrix(Y_val_label, Y_val_pred_label, labels=[0, 1])
    per_class_accuracy_validation = np.diag(C_val.astype(np.float32)) / np.sum(C_val.astype(np.float32), axis=1)
    print("validation:", per_class_accuracy_validation)



    print("----Testing----")

    X_test_2 = np.concatenate((test_positive1, test_negative1), axis=0)
    test_pos_labels = np.ones((test_positive1.shape[0]), dtype=int)
    test_neg_labels = np.zeros((test_negative1.shape[0]), dtype=int)
    Y_test = np.concatenate((test_pos_labels, test_neg_labels), axis=0)
    print(test_pos_labels.shape, test_neg_labels.shape)

    t = time.time()
    Y_test_pred = clf.predict_proba(X_test_2)
    print("time for testing prediction:", time.time()-t)

    Y_test_pred_label = np.zeros((Y_test.shape[0]))
    Y_test_pred_label[Y_test_pred > 0] = 1




    C_test = metrics.confusion_matrix(Y_test, Y_test_pred_label, labels=[0, 1])
    per_class_accuracy_test = np.diag(C_test.astype(np.float32)) / np.sum(C_test.astype(np.float32), axis=1)
    print("test acc before double threshold:", per_class_accuracy_test)


    Y_test_splice = Y_test_pred[:test_positive1.shape[0]]
    Y_test_authentic = Y_test_pred[test_positive1.shape[0]:]

    plt.figure()
    n_p, bins_p, patches_p = plt.hist(x=Y_test_splice, bins='auto', color='b',rwidth=0.9, label="spliced pixels")
    n_n, bins_n, patches_n = plt.hist(x=Y_test_authentic, bins='auto', color='y',rwidth=0.9, label="authentic pixels")

    plt.grid(axis='y')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('histogram of testing predicted Y')
    plt.legend()

    plt.savefig(save_dir_matFiles + "original_resolution/clear_test/histogram_1-1_lab_HierKM_test_weight_rf.png")


    name_loc_prob = pickle.load(open(save_dir_matFiles + 'original_resolution/clear_test/name_loc.pkl', 'rb'))

    counts = np.zeros((len(name_loc_prob), 2))

    for k in range(len(name_loc_prob)):
        counts[k][0] = len(name_loc_prob[k]['spliced_loc'])
        counts[k][1] = len(name_loc_prob[k]['authentic_loc'])

    print("total number of spliced and authentic pixels in testing images:", np.sum(counts, axis=0))

    test_spliced_pred = np.array(Y_test_splice)
    test_authen_pred = np.array(Y_test_authentic)

    counts_splice = counts[:, 0]
    counts_authen = counts[:, 1]

    for k in range(counts.shape[0]):
        splice_pix_num = int(counts[k][0])
        authen_pix_num = int(counts[k][1])
        print(splice_pix_num, authen_pix_num)

        for i in range(splice_pix_num):
            idxi = int(np.sum(counts_splice[:k])) + i
            # print(idxi)
            name_loc_prob[k]['spliced_loc'][i].append(test_spliced_pred[idxi])

        for j in range(authen_pix_num):
            idxj = int(np.sum(counts_authen[:k])) + j
            # print(idxj)
            name_loc_prob[k]['authentic_loc'][j].append(test_authen_pred[idxj])

    with open(save_dir_matFiles + 'original_resolution/clear_test/name_loc_prob_rfw.pkl', 'wb') as fid:
        pickle.dump(name_loc_prob, fid)

    # with open(save_dir_matFiles + 'original_resolution/clear_test/name_loc_prob_rfw.pkl', 'rb') as fid:
    #     name_loc_prob = pickle.load(fid)

    output_prob_map = np.zeros((len(name_loc_prob), 256, 384))
    gt_map = np.zeros((len(name_loc_prob), 256, 384))

    for k in range(len(name_loc_prob)):
        # name_loc_prob[k] is a dictionary
        splice_pixel_loc = name_loc_prob[k]['spliced_loc']  # list
        authen_pixel_loc = name_loc_prob[k]['authentic_loc']  # list
        for pos_pixel in range(len(splice_pixel_loc)):
            i = splice_pixel_loc[pos_pixel][0]
            j = splice_pixel_loc[pos_pixel][1]
            output_prob_map[k, i, j] = splice_pixel_loc[pos_pixel][-1]
            gt_map[k, i, j] = 1

        for neg_pixel in range(len(authen_pixel_loc)):
            i = authen_pixel_loc[neg_pixel][0]
            j = authen_pixel_loc[neg_pixel][1]
            output_prob_map[k, i, j] = authen_pixel_loc[neg_pixel][-1]
            gt_map[k, i, j] = -1

    for k in range(len(name_loc_prob)):
        plt.figure(0)
        plt.imshow(output_prob_map[k], cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.savefig(save_dir_matFiles + 'original_resolution/clear_test/visual/' + name_loc_prob[k]['test_name'][
                                                                    :-4] + '_output_probmap_1-1_lab_HKM_rfw.png')
        plt.close(0)

        # plt.figure(0)
        # plt.imshow(gt_map[k], cmap='coolwarm')
        # plt.colorbar()
        # plt.savefig(save_dir_matFiles + 'columbia/' + name_loc_prob[k]['test_name'][:-4] + 'gt_map.png')


     #%%
    # find where the spike locates
    # spike_loc_map = np.zeros((output_prob_map.shape))
    # for k in range(len(name_loc_prob)):
    #     for i in range(output_prob_map.shape[1]):
    #         for j in range(output_prob_map.shape[2]):
    #             if output_prob_map[k,i,j] > -0.2142 and output_prob_map[k,i,j] <-0.1931 :
    #                 spike_loc_map[k,i,j] = 1
    #
    # for k in range(spike_loc_map.shape[0]):
    #     plt.figure(0)
    #     plt.imshow(spike_loc_map[k])
    #     plt.axis('off')
    #
    #     plt.savefig(save_dir_matFiles + 'original_resolution/clear_test/spike/' + name_loc_prob[k]['test_name'][
    #                                                                                :-4] + '_output_probmap_1-1_lab_HKM_spike.png')
    #     plt.close(0)




