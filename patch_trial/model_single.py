import os
import sys
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from pixelhop2 import Pixelhop2
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score
from skimage.measure import block_reduce
from skimage.util import view_as_windows


def load_data(region):
    print(region)
    # Load Data
    data = np.load(region + '.npz')
    train_images = data['train_images']
    test_images = data['test_images']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    train_names = data['train_names']
    test_names = data['test_names']

    # Subset of the data
    N_train = train_images.shape[0]
    N_test = test_images.shape[0]
    # N_train=1000
    # N_test=500
    np.random.seed(0)
    train_idx = np.random.choice(train_images.shape[0], size=N_train, replace=False)
    np.random.seed(0)
    test_idx = np.random.choice(test_images.shape[0], size=N_test, replace=False)
    train_images = train_images[train_idx]
    train_labels = train_labels[train_idx]
    train_names = train_names[train_idx]
    test_images = test_images[test_idx]
    test_labels = test_labels[test_idx]
    test_names = test_names[test_idx]
    print("Training Image Shape", train_images.shape)
    print("Testing Image Shape", test_images.shape)

    return train_images, train_labels, train_names, test_images, test_labels, test_names


def model(train_images, train_labels, test_images, test_labels, region, save_p2=None, save_clf_list=None,
          save_log_clf=None):
    if not os.path.exists('save'):
        os.makedirs('save')
    depth = 2
    N_train = train_images.shape[0]
    N_test = test_images.shape[0]

    # example callback function for collecting patches and its inverse
    def Shrink(X, shrinkArg, max_pooling=True):
        if max_pooling:
            X = block_reduce(X, (1, 2, 2, 1), np.max)
        win = shrinkArg['win']
        X = view_as_windows(X, (1, win, win, 1))
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
        return X

    # example callback function for how to concate features from different hops
    def Concat(X, concatArg):
        return X

    # set args for PixelHop++
    SaabArgs = [{'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None},
                {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None},
                {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None}]
    shrinkArgs = [{'func': Shrink, 'win': 3},
                  {'func': Shrink, 'win': 3},
                  {'func': Shrink, 'win': 3}]
    concatArg = {'func': Concat}

    # PixlHop ++
    if save_p2 == None:
        print("--- Training ---")
        p2 = Pixelhop2(depth=depth, TH1=0.01, TH2=0.003, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs,
                       concatArg=concatArg).fit(train_images)
        # pickle.dump(p2, open("save/" + region + "_" + "p2_xgb_log.pkl", "wb" ))
    else:
        p2 = save_p2

    print("--- Testing ---")
    train_feature = p2.transform(train_images)
    test_feature = p2.transform(test_images)

    print(train_feature[0].shape, train_feature[1].shape)
    print("------- DONE -------\n")

    # Calculate the Energy
    total_energy = 0
    discared_energy = 0
    for i in range(depth):
        print("Layer ", i)
        print("  # nodes:", len(p2.Energy[i]))
        print('  # intermediate leaf nodes:', sum(p2.Energy[i] >= p2.TH2))
        print('  # discarded leaf nodes:', sum(p2.Energy[i] < p2.TH2))
        print('  Total Energy:', sum(p2.Energy[i][p2.Energy[i] >= p2.TH2]))

    # Select features
    train_feature = train_feature[1]
    test_feature = test_feature[1]
    n_channels = train_feature.shape[-1]
    if save_clf_list == None:
        # Traning
        # XGBoost Parameters
        params = {
            'min_child_weight': [1, 2, 4, 5, 6, 7, 8, 10, 11, 14, 15, 17, 21],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'learning_rate': [0.1, 0.01, 0.2]
        }

        folds = 4
        param_comb = 20
        clf_list = []
        for i in range(n_channels):
            train_flat = train_feature[:, :, :, i].reshape(N_train, -1)
            # XGB
            xgb = XGBClassifier()
            # K-Fold
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
            random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
                                               n_jobs=8, cv=skf.split(train_flat, train_labels), random_state=1001)
            random_search.fit(train_flat, train_labels)
            clf_list.append(random_search.best_estimator_)
        pickle.dump(clf_list, open("save/" + region + "_" + "clf_list.pkl", "wb"))
    else:
        print('load')
        clf_list = save_clf_list

    # Testing
    train_pred_prob = []
    test_pred_prob = []
    for i in range(n_channels):
        train_flat = train_feature[:, :, :, i].reshape(N_train, -1)
        test_flat = test_feature[:, :, :, i].reshape(N_test, -1)
        clf = clf_list[i]
        train_pred_prob.append(clf.predict_proba(train_flat)[:, 1])
        test_pred_prob.append(clf.predict_proba(test_flat)[:, 1])

    train_pred_prob = np.array(train_pred_prob).transpose()
    test_pred_prob = np.array(test_pred_prob).transpose()

    # Final Classifier
    if save_log_clf == None:
        log_clf = LogisticRegression(class_weight='balanced', n_jobs=-1)
        log_clf.fit(train_pred_prob, train_labels)
        pickle.dump(log_clf, open("save/" + region + "_" + "log_clf.pkl", "wb"))
    else:
        print('load')
        log_clf = save_log_clf

    train_prob = log_clf.predict_proba(train_pred_prob)[:, 1]
    test_prob = log_clf.predict_proba(test_pred_prob)[:, 1]

    train_pred = log_clf.predict(train_pred_prob)
    test_pred = log_clf.predict(test_pred_prob)

    print('Frame Train ACC:', accuracy_score(train_labels, train_pred))
    print('Frame Test ACC:', accuracy_score(test_labels, test_pred))

    return train_prob, test_prob


def plot_ROC(fpr, tpr, roc_auc, title='Receiver operating characteristic', filename='ROC.png'):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()


def vid_name(name):
    return name.replace(name.split('_')[-1], '')[0:-1]


def vid_prob(probs, names):
    video = {}
    count = {}
    for name in names:
        video[vid_name(name)] = 0
        count[vid_name(name)] = 0
    for name in names:
        count[vid_name(name)] += 1
    for idx, prob in enumerate(probs):
        video[vid_name(names[idx])] += prob
    gts = []
    vid_probs = []
    for key in video:
        if 'real' in key:
            gts.append(1)
        else:
            gts.append(0)
        video[key] = video[key] / count[key]
        vid_probs.append(video[key])
    return gts, vid_probs


def evaluate(probs, names):
    vid_gts, vid_probs = vid_prob(probs, names)
    fpr, tpr, thresholds = metrics.roc_curve(vid_gts, vid_probs)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc


if __name__ == "__main__":
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    save = False
    start_time = time.time()
    region = sys.argv[1]

    train_images, train_labels, train_names, test_images, test_labels, test_names = load_data(region)
    if save == True:
        train_patch_prob, test_patch_prob = model(train_images, train_labels, test_images, test_labels, region=region,
                                                  save_p2=None,
                                                  save_clf_list=pickle.load(
                                                      open("save/" + region + "_" + "clf_list.pkl", 'rb')),
                                                  save_log_clf=pickle.load(
                                                      open("save/" + region + "_" + "log_clf.pkl", 'rb')))
    else:
        train_patch_prob, test_patch_prob = model(train_images, train_labels, test_images, test_labels, region=region)

    # Training Figure
    fpr, tpr, roc_auc = evaluate(train_patch_prob, train_names)
    print(region + " Video Train AUC:", roc_auc)
    plot_ROC(fpr, tpr, roc_auc, title='Training Receiver operating characteristic',
             filename='Train_ROC_XGB_LOG_' + region + '.png')

    # Testing Figure
    fpr, tpr, roc_auc = evaluate(test_patch_prob, test_names)
    roc_auc = metrics.auc(fpr, tpr)
    print(region + " Video Test AUC:", roc_auc)
    plot_ROC(fpr, tpr, roc_auc, title='Testing Receiver operating characteristic',
             filename='Test_ROC_XGB_LOG_' + region + '.png')

    # Time
    print("--- %s seconds ---" % (time.time() - start_time))
