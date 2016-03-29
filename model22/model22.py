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
#y = y.reshape(y.shape[0],1)
#print y
#print X

X = np.array(X)
y = np.array(y)


print X.shape
print y.shape


from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn import cross_validation
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm

print "Imputation started"
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)


X = imp.transform(X)

print "Transformation started"
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

print "transformation done"

with open('means.pickle', 'w') as f:
    pickle.dump([imp, scaler], f)

sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.5, random_state=42)
print len(sss)

for train_index, test_index in sss:
    X_1, X_2 = X[train_index], X[test_index]
    y_1, y_2 = y[train_index], y[test_index]

#X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

print X_1.shape
print y_1.shape
print X_2.shape
print y_2.shape

print "classifier 1 starting"

clf1 = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, C=1, random_state=0 )
clf1.fit(X_1,y_1)

print "saving classifier 1"
with open('clf1.pickle', 'w') as f:
    pickle.dump(clf1, f)

y_pred_2 = clf1.decision_function(X_2)

np.savetxt("prob_y_pred_2_clf1.csv", y_pred_2, delimiter=',')

print "classifier 2 starting"

clf2 = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, C=1, random_state=0 )
clf2.fit(X_2, y_2)

print "saving classifier 2"

with open('clf2.pickle', 'w') as f:
    pickle.dump(clf2, f)

y_pred_1 = clf2.decision_function(X_1)

np.savetxt("prob_y_pred_1_clf2.csv", y_pred_1, delimiter=',')



print "training on complete data classifier3"

clf3 = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, C=1, random_state=0 )
clf3.fit(X,y)
y_pred_all = clf3.predict(X)
print metrics.accuracy_score(y, y_pred_all)

print "saving classifier  3"


with open('clf3.pickle', 'w') as f:
    pickle.dump(clf3, f)





