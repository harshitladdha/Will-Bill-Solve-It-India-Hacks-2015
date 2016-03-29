import numpy as np
import pandas as pd
from collections import Counter

data = pd.read_csv("result2.csv")
#print data.shape
#print data.columns

data = data.drop(['user_id','problem_id'],1)
#print data.columns

X = data[data.columns[1:]]
y = data[data.columns[0]]
#print y
#print X

X = np.array(X)
y = np.array(y)


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

print X.shape

print "Imputation started"

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)
print X.shape

print "Transformation started"

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
print X.shape

print "transformation done"

with open(r"means2.pickle", "wb") as output_file:
  pickle.dump([imp, scaler], output_file)

print X.shape
from sklearn import cross_validation

print "cross_validation started"

sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.5, random_state=42)
print len(sss)

for train_index, test_index in sss:
    X_1, X_2 = X[train_index], X[test_index]
    y_1, y_2 = y[train_index], y[test_index]


print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.1,
          "max_depth": 10,
          "min_child_weight": 20,
          "silent": 1,
          "subsample": 1,
          "colsample_bytree": 0.6,
          "seed": 1}
params['eval_metric'] = ['auc', 'error']
params['nthread'] = 4
num_round = 201

d_1 = xgb.DMatrix(X_1, y_1)
d_2 = xgb.DMatrix(X_2, y_2)
#watchlist  = [(dtest,'eval'), (dtrain,'train')]

print "starting clf1"

clf1 = xgb.train(params, d_1, num_round)
#print clf

print "saving clf1"

clf1.save_model('xgb1.model')

y_2_pred = clf1.predict(d_2)

np.savetxt("prob_y_pred_2_clf1.csv", y_2_pred, delimiter=',')

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

