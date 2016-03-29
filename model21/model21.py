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

alpha_list = [ 0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
n_iter_list = [5, 10, 15, 20, 25]
alpha = alpha_list[1]
n_iter = n_iter_list[3]



print "alpha = "+str(alpha)
print "n_iter = "+str(n_iter)

print "classifier 1 starting"

clf1 = SGDClassifier(loss="log", penalty="l2", alpha=alpha, n_iter=n_iter, random_state=0)
clf1.fit(X_1,y_1)

print "saving classifier 1"
with open('clf1.pickle', 'w') as f:
    pickle.dump(clf1, f)

y_pred_2 = clf1.predict_proba(X_2)

np.savetxt("prob_y_pred_2_clf1.csv", y_pred_2, delimiter=',')

print "classifier 2 starting"

clf2 = SGDClassifier(loss="log", penalty="l2", alpha=alpha, n_iter=n_iter, random_state=0)
clf2.fit(X_2, y_2)

print "saving classifier 2"

with open('clf2.pickle', 'w') as f:
    pickle.dump(clf2, f)

y_pred_1 = clf2.predict_proba(X_1)

np.savetxt("prob_y_pred_1_clf2.csv", y_pred_1, delimiter=',')



print "training on complete data classifier3"

clf3 = SGDClassifier(loss="log", penalty="l2", alpha=alpha, n_iter=n_iter, random_state=0)
clf3.fit(X,y)

print "saving classifier  3"

with open('clf3.pickle', 'w') as f:
    pickle.dump(clf3, f)





