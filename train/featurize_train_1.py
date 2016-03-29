import numpy as np
import pandas as pd
from collections import Counter

submissions_flat = pd.read_csv("submissions_flat.csv")
users_flat = pd.read_csv("users_flat_train.csv")
problems_flat = pd.read_csv("problems_flat_train.csv")

print submissions_flat.columns
print submissions_flat.shape

print users_flat.columns
print users_flat.shape

print problems_flat.columns
print problems_flat.shape

users_flat = users_flat.drop(['skills','user_type'],1)
problems_flat = problems_flat.drop(['level','tag1','tag2','tag3','tag4','tag5'],1)


print users_flat.columns
print users_flat.shape

print problems_flat.columns
print problems_flat.shape


result = pd.merge(submissions_flat, users_flat, how='left', on='user_id')
#result.to_csv("result.csv", index=False)

result2 = pd.merge(result, problems_flat, how='left', on='problem_id')
result2.to_csv("result2.csv", index=False)