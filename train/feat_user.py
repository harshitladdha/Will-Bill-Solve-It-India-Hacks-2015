import numpy as np
import pandas as pd
from collections import Counter

users = pd.read_csv("users.csv")
print users.shape
print users.columns

"""
skils = []
for i in range(users.shape[0]):
	string = str(users['skills'][i])
	string = string.lower()
	string = string.replace(" ", "")
	skill_ = string.split('|')
	skils = skils + skill_

print len(skils)
c = Counter(skils)
print len(c.items())
for key, item in c.most_common():
	print ("\'"+str(key)+"\',")
	#print '%s, %d' % (key, item)
"""


skills_list = ['c','c++','java','python','c#','php','ruby','javascript','perl','objective-c','javascript(node.js)','javascript(rhino)','python3','go','text','haskell','clojure','scala','befunge','rust','pascal','r(rscript)','lisp','c++(g++4.8.1)','java(openjdk1.7.0_09)']

for item in skills_list:
	users[item] = 0

print users['c']

print "doing skills............"

for i in range(users.shape[0]):
	if i%100==0:
		print i
	string = str(users['skills'][i])
	string = string.lower()
	string = string.replace(" ", "")
	skill_ = string.split('|')

	for item in skill_ :
		if item in skills_list:
			users[item][i] = 1

print "doing user_type..........."
users['S'] = 0
users['W'] = 0

for i in range(users.shape[0]):
	if i%100==0:
		print i
	if users['user_type'][i] == 'W':
		users['W'][i] = 1
	elif users['user_type'][i] == 'S':
		users['S'][i] = 1

users.to_csv("users_flat_train.csv", index=False)
