from tensorflow.keras.datasets import cifar10
import numpy as np
import pickle
from sklearn import svm

np.random.seed(20)	

f = open('x_test_trans_single.pkl', 'rb')
x_test_trans = pickle.load(f)
f.close()

f = open('x_train_trans_single.pkl', 'rb')
x_train_trans = pickle.load(f)
f.close()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# rfclf = RandomForestClassifier(n_estimators=100, random_state=4)
# rfclf.fit(x_train_trans, np.squeeze(y_train))	
# score = rfclf.score(x_test_trans, np.squeeze(y_test))
# print('Predicted score =', score)
# score_tr = rfclf.score(x_train_trans, np.squeeze(y_train))	
# print('Train score =', score_tr)


svmclf = svm.SVC(C=2, kernel='rbf', gamma=12)
svmclf.fit(x_train_trans, np.squeeze(y_train))
score = svmclf.score(x_test_trans, np.squeeze(y_test))
print('Predicted score =', score)
# score_tr = svmclf.score(x_train_trans, np.squeeze(y_train))
# print('Train score =', score_tr)
