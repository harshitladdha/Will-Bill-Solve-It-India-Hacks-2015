import numpy as np
import pandas as pd
from collections import Counter
import math


data  = pd.read_csv("result2_test.csv")
submit = data['Id']

model19 = np.genfromtxt("base/prob_test_model19.csv", delimiter=',')
model19 = model19[:,1]
model19 = model19.reshape(model19.shape[0],1)
print model19.shape

model20 = np.genfromtxt("base/prob_test_model20.csv", delimiter=',')
model20 = model20[:,1]
model20 = model20.reshape(model20.shape[0],1)
print model20.shape

model21 = np.genfromtxt("base/prob_test_model21.csv", delimiter=',')
model21 = model21[:,1]
model21 = model21.reshape(model21.shape[0],1)
print model21.shape

model22 = np.genfromtxt("base/prob_test_model23.csv", delimiter=',')
#model22 = model22[:,1]
model22 = model22.reshape(model22.shape[0],1)
print model22.shape

model26 = np.genfromtxt("stage2/prob_test_stage2_model26.csv", delimiter=',')
#model26 = model26[:,1]
model26 = model26.reshape(model26.shape[0],1)
print model26.shape

model27 = np.genfromtxt("stage2/prob_test_stage2_model27.csv", delimiter=',')
#model27 = model27[:,1]
model27 = model27.reshape(model27.shape[0],1)
print model27.shape

model28 = np.genfromtxt("stage2/prob_test_stage2_model28.csv", delimiter=',')
#model28 = model28[:,1]
model28 = model28.reshape(model28.shape[0],1)
print model28.shape

model29 = np.genfromtxt("stage2/prob_test_stage2_model29.csv", delimiter=',')
#model29 = model29[:,1]
model29 = model29.reshape(model29.shape[0],1)
print model29.shape

#y_pred1 = (model19+model22*2)/3
y_pred1 = np.power(model19, 0.33)*np.power(model22, 0.67)
#y_pred2 = (model22*2+model27+model28)/4
y_pred2 = np.power(model22, 0.5)*np.power(model27, 0.25)*np.power(model28, 0.25)
#y_pred = (y_pred1+y_pred2)/2

y_pred = np.power(y_pred1,0.5)*np.power(y_pred2, 0.5)




prob_thresh = 0.74
#print prob_thresh
predict_thresh = y_pred>prob_thresh
predict_thresh = predict_thresh+0
print predict_thresh.sum(), prob_thresh
predict_thresh_df = pd.DataFrame(predict_thresh,columns=['solved_status'])
#print predict_thresh_df.shape
submit_new = pd.concat([submit,predict_thresh_df], axis=1)
submit_new.to_csv("submit_test_stage2_"+str(prob_thresh)+".csv", index=False)


