import numpy as np
import pandas as pd
from collections import Counter

data = pd.read_csv("result2.csv")
data = data.drop(['user_id','problem_id'],1)
#print data.columns

X = data[data.columns[1:]]
y = data[data.columns[0]]
#print y
#print X

X = np.array(X)
y = np.array(y)

print "total x"
print X.shape
print "total y"
print y.shape
from sklearn.cross_validation import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.5, random_state=42)
print len(sss)

for train_index, test_index in sss:
    X_1, X_2 = X[train_index], X[test_index]
    y_1, y_2 = y[train_index], y[test_index]

print "X_1"
print X_1.shape
print "X_2"
print X_2.shape
print "y_1"
print y_1.shape
print "y_2"
print y_2.shape

y_new = y_1


model19 = np.genfromtxt("prob_y_pred_1_clf2_model19.csv", delimiter=',')
model19 = model19[:,1]
model19 = model19.reshape(model19.shape[0],1)
print model19.shape

model20 = np.genfromtxt("prob_y_pred_1_clf2_model20.csv", delimiter=',')
model20 = model20[:,1]
model20 = model20.reshape(model20.shape[0],1)
print model20.shape

model21 = np.genfromtxt("prob_y_pred_1_clf2_model21.csv", delimiter=',')
model21 = model21[:,1]
model21 = model21.reshape(model21.shape[0],1)
print model21.shape

model22 = np.genfromtxt("prob_y_pred_1_clf2_model22.csv", delimiter=',')
#model22 = model22[:,1]
model22 = model22.reshape(model22.shape[0],1)
print model22.shape

model23 = np.genfromtxt("prob_y_pred_1_clf2_model23.csv", delimiter=',')
#model23 = model23[:,1]
model23 = model22.reshape(model23.shape[0],1)
print model23.shape

X_new = model19
X_new = np.append(X_new, model20, 1)
X_new = np.append(X_new, model21, 1)
X_new = np.append(X_new, model22, 1)
X_new = np.append(X_new, model23, 1)
print "X_new"
print X_new.shape
print "y_new"
print y_new.shape





from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import cross_validation



print "Transformation started"

scaler = preprocessing.StandardScaler().fit(X_new)
X_new = scaler.transform(X_new)
print X_new.shape

print "transformation done"

with open(r"means2.pickle", "wb") as output_file:
  pickle.dump(scaler, output_file)


sss_new = StratifiedShuffleSplit(y_new, n_iter=1, test_size=0.33, random_state=42)
print len(sss_new)

for train_index, test_index in sss_new:
    X_new_train, X_new_test = X_new[train_index], X_new[test_index]
    y_new_train, y_new_test = y_new[train_index], y_new[test_index]

print X_new_train.shape
print X_new_test.shape
print y_new_train.shape
print y_new_test.shape


print "cross_validation started"
#max_depth_list = [5, 4, 3, 2]
#max_depth_list = [3]
#min_child_weight_list = [20, 40, 60 ,80]
#min_child_weight_list = [200]

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.1,
          "max_depth": 3,
          "min_child_weight": 200,
          "silent": 1,
          "subsample": 1,
          "colsample_bytree": 1,
          "seed": 1}
params['eval_metric'] = ['auc', 'error']
params['nthread'] = 4
num_round = 42
"""
dtrain = xgb.DMatrix(X_new_train, y_new_train)
dtest = xgb.DMatrix(X_new_test, y_new_test)
watchlist  = [(dtest,'eval'), (dtrain,'train')]

print "starting clf1"

clf = xgb.train(params, dtrain, num_round, watchlist)
#print clf

#print "saving clf1"

#clf.save_model('xgb1.model')


print "Train error"
y_new_train_preds = clf.predict(dtrain)
preds = y_new_train_preds
labels = dtrain.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))


print "Test error"
y_new_test_preds = clf.predict(dtest)
np.savetxt("prob_random.csv", y_new_test_preds, delimiter=',')
preds = y_new_test_preds
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
"""

dtrain_tot = xgb.DMatrix(X_new, y_new)
watchlist_tot  = [(dtrain_tot,'train')]
clf2 = xgb.train(params, dtrain_tot, num_round, watchlist_tot)
print "saving clf2"

clf2.save_model('xgb.model')


#np.savetxt("prob_y_pred_2_clf1.csv", y_2_pred, delimiter=',')
"""
print "starting clf2"

clf2 = xgb.train(params, d_2, num_round)
#print clf

print "saving clf2"

clf2.save_model('xgb2.model')

y_1_pred = clf2.predict(d_1)

np.savetxt("prob_y_pred_1_clf2.csv", y_1_pred, delimiter=',')

d_total = xgb.DMatrix(X, y)

print "starting clf3"

clf3 = xgb.train(params, d_total, num_round)
#print clf

print "saving clf3"
clf3.save_model('xgb3.model')

preds = clf3.predict(d_total)
labels = d_total.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
"""
