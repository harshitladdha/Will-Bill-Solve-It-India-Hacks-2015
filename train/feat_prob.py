import numpy as np
import pandas as pd
from collections import Counter

problems = pd.read_csv("problems.csv")
print problems.shape
print problems.columns
problems['level_factor'] = np.nan
print problems.shape
#print problems

print "doing levels......"

for i in range(problems.shape[0]):
	if i%100 == 0:
		print i
	if problems['level'][i] == 'E':
		problems['level_factor'][i] = 0

	elif problems['level'][i] == 'E-M':
		problems['level_factor'][i] = 0.5

	elif problems['level'][i] == 'M':
		problems['level_factor'][i] = 1

	elif problems['level'][i] == 'M-H':
		problems['level_factor'][i] = 2

	elif problems['level'][i] == 'H':
		problems['level_factor'][i] = 3

#print problems

tags = ['Algorithms','Math','Ad-Hoc','Dynamic Programming','Basic Programming','Data Structures','Implementation','Sorting','Graph Theory','Greedy','Combinatorics','Number Theory','Binary Search','Hashing','DFS','Brute Force','String Algorithms','Recursion','BIT','Geometry','Bit manipulation','Segment Trees','Probability','Matrix Exponentiation','BFS','Bitmask','Disjoint Set','Trees','Game Theory','HashMap','Sieve','Priority Queue','Fenwick Tree','Minimum Spanning Tree','Heap','Stack','Simple-math','Modular exponentiation','Modular arithmetic','Prime Factorization','Binary Search Tree','Dijkstra','Shortest-path','Two-pointer','Simulation','Divide And Conquer','Trie','Sqrt-Decomposition','Floyd Warshall','Sailesh Arya','KMP','Memoization','Suffix Arrays','Heavy light decomposition','Binary Tree','adhoc','Matching','Ad-hoc','Queue','Expectation','Data-Structures','Extended Euclid','Completed','Priority-Queue','String-Manipulation','Basic-Programming','Set','GCD','Very Easy','Bellman Ford','Flow','cake-walk','Easy-medium','FFT','Line-sweep','Kruskal','Bipartite Graph','Maps']

for item in tags:
	problems[item] = 0

print problems['Algorithms']

print "doing tags....."

for i in range(problems.shape[0]):
	if i%100 == 0:
		print i
	if str(problems['tag1'][i]) in tags:
		#print str(problems['tag1'][i])
		problems[problems['tag1'][i]][i] = 1

	if str(problems['tag2'][i]) in tags:
		#print str(problems['tag1'][i])
		problems[problems['tag2'][i]][i] = 1

	if str(problems['tag3'][i]) in tags:
		#print str(problems['tag1'][i])
		problems[problems['tag3'][i]][i] = 1

	if str(problems['tag4'][i]) in tags:
		#print str(problems['tag1'][i])
		problems[problems['tag4'][i]][i] = 1

	if str(problems['tag5'][i]) in tags:
		#print str(problems['tag1'][i])
		problems[problems['tag5'][i]][i] = 1
print problems['Algorithms']


problems.to_csv("problems_flat_train.csv", index=False)
