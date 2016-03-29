#write code for generating pred 1, pred 2, prob_test using classifier and copying these into model19

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





with cd("model19_withclassifier"):
	print(datetime.now(),"-+- Starting predict_train.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict_train.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict_train.py\n")

	print(datetime.now(),"-+- Starting predict.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict.py\n")
