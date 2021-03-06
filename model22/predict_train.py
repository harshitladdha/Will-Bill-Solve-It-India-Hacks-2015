import numpy as np
import pandas as pd
from collections import Counter

data  = pd.read_csv("result2.csv")


data = data.drop(['user_id','problem_id','solved_status'],1)
print data.columns
print data.shape

X = data
#print y
#print X

X = np.array(X)



from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
#import cPickle as pickle
import pickle

from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score

with open('trained.pickle') as f:
    imp, scaler, clf = pickle.load(f)


print "Imputation started"

X = imp.transform(X)

print "Transformation started"
X = scaler.transform(X)

print "transformation done"

#y_pred_class = clf.predict(X)
y_pred = clf.predict_proba(X)
#y_pred_df = pd.DataFrame(y_pred_class,columns=['solved_status'])
#submit = pd.concat([submit,y_pred_df], axis=1)
#submit.to_csv("submit_test_new_classes.csv", index=False)

np.savetxt("prob_train.csv", y_pred, delimiter=',')
"""

y_pred = np.genfromtxt("prob_new_classes.csv", delimiter=',')
print y_pred.shape
y_pred = y_pred[:,1]
print y_pred.shape
print y_pred

prob_thresh = 0.4
predict_thresh = y_pred>prob_thresh
predict_thresh = predict_thresh+0
print predict_thresh
predict_thresh_df = pd.DataFrame(predict_thresh,columns=['solved_status'])
print predict_thresh_df.shape
submit = pd.concat([submit,predict_thresh_df], axis=1)
submit.to_csv("submit_test_new_classes_"+str(prob_thresh)+".csv", index=False)
"""
