import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

pos_sample_test = np.squeeze(np.load(r"/mnt/zhengwen/new_trial/test_ori_sing_output_2.npy"))
neg_sample_test = np.squeeze(np.load(r"/mnt/zhengwen/new_trial/test_ste_sing_output_2.npy"))

pos_sample_test = pos_sample_test.reshape((pos_sample_test.shape[0], -1))
neg_sample_test = neg_sample_test.reshape((neg_sample_test.shape[0], -1))

print(pos_sample_test.shape)
print(neg_sample_test.shape)



label_pos = np.ones(len(pos_sample_test))
label_neg = -np.ones(len(neg_sample_test))

sample_test = np.concatenate((pos_sample_test, neg_sample_test), axis=0)
label_test = np.concatenate((label_pos, label_neg), axis=0)

# idx = np.random.permutation(len(label_test))
# sample_test = sample_test[idx]
# label_test = label_test[idx]

f = open('/mnt/zhengwen/new_trial/RMclf_single_p2.pkl', 'rb')
clf = pickle.load(f)
f.close()

print(clf.score(sample_test, label_test))
