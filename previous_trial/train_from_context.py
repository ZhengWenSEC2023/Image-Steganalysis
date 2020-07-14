import numpy as np
import os
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pickle
from previous_trial.config_img_stag import config
from previous_trial.Hierarchical_kmeans import HierKmeans
from sklearn.linear_model import LinearRegression

"""
Train a classifier from the extracted context
"""



save_dir_matFiles = config['save_dir_matFiles']
window_rad = config['wind_rad']
thres_bin_edges = config['thres_bin_edges']
ori_image_dir = config['ori_image_dir']
uerf_image_dir = config['uerf_image_dir']
context_dir = config['context_dir']
train_ori_context_name = config['train_ori_context_name']
weight_root_name = config['weight_root_name']
train_pos_target_name = config['train_pos_target_name']
train_neg_target_name = config['train_neg_target_name']
train_pos_contxt_name = config['train_pos_contxt_name']
train_neg_contxt_name = config['train_neg_contxt_name']
test_ori_contxt_name = config['test_ori_contxt_name']
test_contxt_name = config['test_contxt_name']
regressor_dir = config['regressor_dir']
regressor_name = config['regressor_name']
plot_dir = config['plot_dir']
one_hop_stp_lst_name = config['one_hop_stp_lst_name']

print('Begin to load data')
start = time.time()
pos_target = np.load(os.path.join(save_dir_matFiles, context_dir, train_pos_target_name))
neg_target = np.load(os.path.join(save_dir_matFiles, context_dir, train_neg_target_name))

pos_context = np.load(os.path.join(save_dir_matFiles, context_dir, train_pos_contxt_name))
neg_context = np.load(os.path.join(save_dir_matFiles, context_dir, train_neg_contxt_name))

# pos_context = pos_context[:500000]
# neg_context = neg_context[:500000]
# pos_target =  pos_target[:500000]
# neg_target =  neg_target[:500000]

train_context = np.concatenate((pos_context, neg_context), axis=0)
train_target = np.concatenate((pos_target, neg_target), axis=0)
end = time.time()

print('End of load data')
print("Time escaped", (end - start))

print(train_context.shape)
print(train_target.shape)

metric = {'min_num_sample': 100,
          'purity': 0.9}
regr = HierKmeans(depth=25, learner=LinearRegression(), num_cluster=2, metric=metric)

print('Begin to fit RF regressor')
regr.fit(train_context, train_target)
print(regr.nodes.keys())
print('Fit finished')


# print('Begin to fit RF regressor')
# regr = RandomForestRegressor(n_estimators=300)
# regr.fit(train_context, train_target)
# print('Fit finished')
if not os.path.isdir(os.path.join(save_dir_matFiles, regressor_dir)):
    os.makedirs(os.path.join(save_dir_matFiles, regressor_dir))
pkl = open(os.path.join(save_dir_matFiles, regressor_dir, regressor_name), 'wb')
pickle.dump(regr, pkl)
pkl.close()

"""
Val: percentage of positive class and negative class
"""

print("Begin to predict")
# pos_result = regr.predict_proba(pos_context)
# neg_result = regr.predict_proba(neg_context)
pos_result = regr.predict_proba(pos_context)
neg_result = regr.predict_proba(neg_context)
print("End of predict")

pos_pos_percentage = np.sum(pos_result > 0) / np.sum(pos_result != 0)
pos_neg_percentage = np.sum(pos_result < 0) / np.sum(pos_result != 0)
neg_pos_percentage = np.sum(neg_result > 0) / np.sum(neg_result != 0)
neg_neg_percentage = np.sum(neg_result < 0) / np.sum(neg_result != 0)

print("pos result", pos_pos_percentage, '%positive')
print("pos result", pos_neg_percentage, '%negative')
print("neg result", neg_pos_percentage, '%positive')
print("neg result", neg_neg_percentage, '%negative')

if not os.path.isdir(os.path.join(save_dir_matFiles, plot_dir)):
    os.makedirs(os.path.join(save_dir_matFiles, plot_dir))
plt.hist(pos_result, bins=50, color=sns.desaturate("indianred", .8), alpha=.4)
plt.title("positive distribution")
plt.savefig(os.path.join(save_dir_matFiles, plot_dir, 'distribution_train_pos.png'))

plt.hist(neg_result, bins=50, color=sns.desaturate("indianred", .8), alpha=.4)
plt.title("negative distribution")
plt.savefig(os.path.join(save_dir_matFiles, plot_dir, 'distribution_train_neg.png'))