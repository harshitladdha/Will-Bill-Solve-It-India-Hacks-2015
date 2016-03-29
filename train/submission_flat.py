import numpy as np
import pandas as pd
from collections import Counter

submissions = pd.read_csv("submissions.csv")


#print submissions.shape
#print submissions.columns

submissions = submissions.drop(['result','language_used','execution_time'],1)
#print submissions.shape
#print submissions.columns


"""
list_solved = submissions['solved_status'].tolist()
l = Counter(list_solved)
print l.items()
for key, item in l.most_common():
	#print ("\'"+str(key)+"\',")
	print '\'%s\',' % (key)

users = submissions['user_id'].tolist()
users_set = set(users)
users = list(users_set)

problems = submissions['problem_id'].tolist()
problems_set = set(problems)
problems = list(problems_set)

print len(users)
print len(problems)
"""

submissions = submissions[submissions.solved_status != 'UK']
#print submissions.shape

"""
for i in range(submissions.shape[0]):
	if i%100==0:
		print i
	if submissions['solved_status'][i] == 'SO':
		submissions['solved_status'][i] = 1
	elif submissions['solved_status'][i] == 'AT':
		submissions['solved_status'][i] = 0
	else:
		print submissions['solved_status'][i]

submissions.to_csv("submissions_01.csv", index=False)
"""

submissions['solved_status'] = submissions['solved_status']=='SO'
submissions['solved_status'] = submissions['solved_status']+0
#submissions.to_csv("submissions_01.csv", index=False)


print "deleting duplicates"
submissionsnew = submissions.groupby(['user_id','problem_id'], group_keys=False).apply(lambda x: x.ix[x.solved_status.idxmax()])
submissionsnew.to_csv("submissions_flat.csv", index=False)

"""
#print "removing duplicates"                                    
submissionsnew = submissions.drop_duplicates(subset=['user_id','problem_id'], keep='last')
print submissionsnew.shape
#submissionsnew.to_csv("submissions_flat.csv", index=False)

a = submissionsnew['solved_status']=='AT'
print type(a)
print a.shape
a=a+0
print a


a = a.tolist()
print type(a)
print len(a)
a = a+1
print a
"""
"""
print "correcting duplicates"
for i in range(submissionsnew.shape[0]):
	print i
	if a[i] :
		print "not solved"
		print submissionsnew['user_id'].shape
		userid = submissionsnew['user_id'][i]r4     
		problemid = submissionsnew['problem_id'][i]
		subm = submissions[submissions.user_id == userid and submissions.problem_id == problemid]
		print subm

"""








