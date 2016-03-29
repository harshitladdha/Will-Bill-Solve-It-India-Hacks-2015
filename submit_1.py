from os import system
from time import sleep
from datetime import datetime
import subprocess
import os
from shutil import copyfile


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)



print(datetime.now(),"-+- Starting feat_prob.py")

with cd("train"):
	#subprocess.call("ls")
	
	print(datetime.now(),"-+- Starting feat_prob.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python feat_prob.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done feat_prob.py\n")


	print(datetime.now(),"-+- Starting feat_user.py")

	#	IN  :::  users.csv
	#   OUT :::  users_flat_train.csv
	#

	#feat_user.py
	system('python feat_user.py')
	print(datetime.now(),"-+- Done feat_user.py\n")
	

	#	IN  :::  submissions.csv
	#   OUT :::  submissions_flat.csv
	#
	print(datetime.now(),"-+- Starting submission_flat.py")
	#submission_flat.py
	system('python submission_flat.py')
	print(datetime.now(),"-+- Done submission_flat.py\n")


	print(datetime.now(),"-+- Starting featurize_train_1.py")
	#	IN  :::  problems_flat_train.csv, users_flat_train.csv, submissions_flat.csv
	#   OUT :::  result2.csv
	#
	#featurize_train_1.py
	system('python featurize_train_1.py')
	print(datetime.now(),"-+- Done featurize_train_1.py")



with cd("test"):

	print(datetime.now(),"-+- Starting feat_prob.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_test.csv
	#

	system('python feat_prob.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done feat_prob.py\n")


	print(datetime.now(),"-+- Starting feat_user.py")

	#	IN  :::  users.csv
	#   OUT :::  users_flat_test.csv
	#

	#feat_user.py
	system('python feat_user.py')
	print(datetime.now(),"-+- Done feat_user.py\n")


	print(datetime.now(),"-+- Starting featurize_test.py")

	#	IN  :::  problems_flat_test.csv, users_flat_test.csv, test.csv
	#   OUT :::  result2_test.csv.csv
	#

	#feat_user.py
	system('python featurize_test.py')
	print(datetime.now(),"-+- Done featurize_test.py \n")

copyfile("train/result2.csv", "model19_withclassifier/result2.csv")
copyfile("test/result2_test.csv", "model19_withclassifier/result2_test.csv")

copyfile("train/result2.csv", "model19/result2.csv")
copyfile("test/result2_test.csv", "model19/result2_test.csv")

copyfile("train/result2.csv", "model20/result2.csv")
copyfile("test/result2_test.csv", "model20/result2_test.csv")

copyfile("train/result2.csv", "model21/result2.csv")
copyfile("test/result2_test.csv", "model21/result2_test.csv")

copyfile("train/result2.csv", "model22/result2.csv")
copyfile("test/result2_test.csv", "model22/result2_test.csv")

copyfile("train/result2.csv", "model23/result2.csv")
copyfile("test/result2_test.csv", "model23/result2_test.csv")

copyfile("train/result2.csv", "model26/result2.csv")
copyfile("test/result2_test.csv", "model26/result2_test.csv")

copyfile("train/result2.csv", "model27/result2.csv")
copyfile("test/result2_test.csv", "model27/result2_test.csv")

copyfile("train/result2.csv", "model28/result2.csv")
copyfile("test/result2_test.csv", "model28/result2_test.csv")

copyfile("train/result2.csv", "model29/result2.csv")
copyfile("test/result2_test.csv", "model29/result2_test.csv")

copyfile("train/result2.csv", "all_test_pred/result2.csv")
copyfile("test/result2_test.csv", "all_test_pred/result2_test.csv")

