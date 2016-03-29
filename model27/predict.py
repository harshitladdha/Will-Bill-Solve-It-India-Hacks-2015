import numpy as np
import pandas as pd
from collections import Counter

data  = pd.read_csv("result2_test.csv")
submit = data['Id']

data = data.drop(['Id','user_id','problem_id'],1)

X = data
X = np.array(X)

model19 = np.genfromtxt("prob_test_model19.csv", delimiter=',')
model19 = model19[:,1]
model19 = model19.reshape(model19.shape[0],1)
print model19.shape

model20 = np.genfromtxt("prob_test_model20.csv", delimiter=',')
model20 = model20[:,1]
model20 = model20.reshape(model20.shape[0],1)
print model20.shape

model21 = np.genfromtxt("prob_test_model21.csv", delimiter=',')
model21 = model21[:,1]
model21 = model21.reshape(model21.shape[0],1)
print model21.shape

model22 = np.genfromtxt("df_y_pred_model22.csv", delimiter=',')
model22 = model22.reshape(model22.shape[0],1)
print model22.shape

model23 = np.genfromtxt("prob_test_model23.csv", delimiter=',')
model23 = model20.reshape(model20.shape[0],1)
print model23.shape

X = np.append(X, model19, 1)
X = np.append(X, model20, 1)
X = np.append(X, model21, 1)
X = np.append(X, model22, 1)
X = np.append(X, model23, 1)
print "X"
print X.shape



from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
#import cPickle as pickle
import pickle

from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

with open('means2.pickle') as f:
	imp, scaler = pickle.load(f)


print "Imputation started"

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)
print X.shape

print "Transformation started"
X = scaler.transform(X)
print "transformation done"


clf = xgb.Booster(model_file='xgb.model')
dtest = xgb.DMatrix(X)
y_pred = clf.predict(dtest)
np.savetxt("prob_test_stage2.csv", y_pred, delimiter=',')

"""
y_pred = np.genfromtxt("prob_test_stage2.csv", delimiter=',')
print y_pred.shape
prob_thresh = 0.74
predict_thresh = y_pred>prob_thresh
predict_thresh = predict_thresh+0
print predict_thresh.sum()
predict_thresh_df = pd.DataFrame(predict_thresh,columns=['solved_status'])
print predict_thresh_df.shape
submit = pd.concat([submit,predict_thresh_df], axis=1)
submit.to_csv("submit_test_stage2_"+str(prob_thresh)+".csv", index=False)
"""