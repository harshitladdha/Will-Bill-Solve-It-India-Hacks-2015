import numpy as np
import pandas as pd
from collections import Counter

data  = pd.read_csv("result2_test.csv")
submit = data['Id']

data = data.drop(['Id','user_id','problem_id'],1)
print data.columns
print data.shape
X = data
X = np.array(X)
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
X = imp.transform(X)
print "Transformation started"
X = scaler.transform(X)
print "transformation done"
clf = xgb.Booster(model_file='xgb3.model')
dtest = xgb.DMatrix(X)
y_pred = clf.predict(dtest)
np.savetxt("prob_test.csv", y_pred, delimiter=',')

"""
y_pred = np.genfromtxt("prob_test.csv", delimiter=',')
print y_pred.shape
prob_thresh = 0.8
predict_thresh = y_pred>prob_thresh
predict_thresh = predict_thresh+0
print predict_thresh.sum()
predict_thresh_df = pd.DataFrame(predict_thresh,columns=['solved_status'])
print predict_thresh_df.shape
submit = pd.concat([submit,predict_thresh_df], axis=1)
#submit.to_csv("submit_test_"+str(prob_thresh)+".csv", index=False)
"""