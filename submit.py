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


with cd("model19"):
	print(datetime.now(),"-+- Starting model19.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python model19.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done model19.py\n")

	print(datetime.now(),"-+- Starting predict.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict.py\n")



with cd("model20"):
	print(datetime.now(),"-+- Starting model20.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python model20.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done model20.py\n")

	print(datetime.now(),"-+- Starting predict.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict.py\n")


with cd("model21"):
	print(datetime.now(),"-+- Starting model21.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python model21.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done model21.py\n")

	print(datetime.now(),"-+- Starting predict.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict.py\n")



with cd("model22"):
	print(datetime.now(),"-+- Starting model22.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python model22.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done model22.py\n")

	print(datetime.now(),"-+- Starting predict.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict.py\n")


with cd("model23"):
	print(datetime.now(),"-+- Starting model23.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python model23.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done model23.py\n")

	print(datetime.now(),"-+- Starting predict.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict.py\n")


copyfile("model19/prob_y_pred_2_clf1.csv", "model26/prob_y_pred_2_clf1_model19.csv")
copyfile("model20/prob_y_pred_2_clf1.csv", "model26/prob_y_pred_2_clf1_model20.csv")
copyfile("model21/prob_y_pred_2_clf1.csv", "model26/prob_y_pred_2_clf1_model21.csv")
copyfile("model22/prob_y_pred_2_clf1.csv", "model26/prob_y_pred_2_clf1_model22.csv")
copyfile("model23/prob_y_pred_2_clf1.csv", "model26/prob_y_pred_2_clf1_model23.csv")

copyfile("model19/prob_test.csv", "model26/prob_test_model19.csv")
copyfile("model20/prob_test.csv", "model26/prob_test_model20.csv")
copyfile("model21/prob_test.csv", "model26/prob_test_model21.csv")
copyfile("model23/prob_test.csv", "model26/prob_test_model23.csv")
copyfile("model22/test_df.csv", "model26/df_y_pred_model22.csv")


copyfile("model19/prob_y_pred_2_clf1.csv", "model27/prob_y_pred_2_clf1_model19.csv")
copyfile("model20/prob_y_pred_2_clf1.csv", "model27/prob_y_pred_2_clf1_model20.csv")
copyfile("model21/prob_y_pred_2_clf1.csv", "model27/prob_y_pred_2_clf1_model21.csv")
copyfile("model22/prob_y_pred_2_clf1.csv", "model27/prob_y_pred_2_clf1_model22.csv")
copyfile("model23/prob_y_pred_2_clf1.csv", "model27/prob_y_pred_2_clf1_model23.csv")

copyfile("model19/prob_test.csv", "model27/prob_test_model19.csv")
copyfile("model20/prob_test.csv", "model27/prob_test_model20.csv")
copyfile("model21/prob_test.csv", "model27/prob_test_model21.csv")
copyfile("model23/prob_test.csv", "model27/prob_test_model23.csv")
copyfile("model22/test_df.csv", "model27/df_y_pred_model22.csv")



copyfile("model19/prob_y_pred_1_clf2.csv", "model28/prob_y_pred_1_clf2_model19.csv")
copyfile("model20/prob_y_pred_1_clf2.csv", "model28/prob_y_pred_1_clf2_model20.csv")
copyfile("model21/prob_y_pred_1_clf2.csv", "model28/prob_y_pred_1_clf2_model21.csv")
copyfile("model22/prob_y_pred_1_clf2.csv", "model28/prob_y_pred_1_clf2_model22.csv")
copyfile("model23/prob_y_pred_1_clf2.csv", "model28/prob_y_pred_1_clf2_model23.csv")

copyfile("model19/prob_test.csv", "model28/prob_test_model19.csv")
copyfile("model20/prob_test.csv", "model28/prob_test_model20.csv")
copyfile("model21/prob_test.csv", "model28/prob_test_model21.csv")
copyfile("model23/prob_test.csv", "model28/prob_test_model23.csv")
copyfile("model22/test_df.csv", "model28/df_y_pred_model22.csv")


copyfile("model19/prob_y_pred_1_clf2.csv", "model29/prob_y_pred_1_clf2_model19.csv")
copyfile("model20/prob_y_pred_1_clf2.csv", "model29/prob_y_pred_1_clf2_model20.csv")
copyfile("model21/prob_y_pred_1_clf2.csv", "model29/prob_y_pred_1_clf2_model21.csv")
copyfile("model22/prob_y_pred_1_clf2.csv", "model29/prob_y_pred_1_clf2_model22.csv")
copyfile("model23/prob_y_pred_1_clf2.csv", "model29/prob_y_pred_1_clf2_model23.csv")

copyfile("model19/prob_test.csv", "model29/prob_test_model19.csv")
copyfile("model20/prob_test.csv", "model29/prob_test_model20.csv")
copyfile("model21/prob_test.csv", "model29/prob_test_model21.csv")
copyfile("model23/prob_test.csv", "model29/prob_test_model23.csv")
copyfile("model22/test_df.csv", "model29/df_y_pred_model22.csv")


with cd("model26"):
	print(datetime.now(),"-+- Starting model26.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python model26.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done model26.py\n")

	print(datetime.now(),"-+- Starting predict.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict.py\n")


with cd("model27"):
	print(datetime.now(),"-+- Starting model27.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python model27.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done model27.py\n")

	print(datetime.now(),"-+- Starting predict.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict.py\n")


with cd("model28"):
	print(datetime.now(),"-+- Starting model28.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python model28.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done model28.py\n")

	print(datetime.now(),"-+- Starting predict.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict.py\n")


with cd("model29"):
	print(datetime.now(),"-+- Starting model29.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python model29.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done model29.py\n")

	print(datetime.now(),"-+- Starting predict.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python predict.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done predict.py\n")





copyfile("model19/prob_test.csv", "all_test_pred/base/prob_test_model19.csv")
copyfile("model20/prob_test.csv", "all_test_pred/base/prob_test_model20.csv")
copyfile("model21/prob_test.csv", "all_test_pred/base/prob_test_model21.csv")
copyfile("model23/prob_test.csv", "all_test_pred/base/prob_test_model23.csv")
copyfile("model22/test_df.csv", "all_test_pred/base/df_y_pred_model22.csv")

copyfile("model26/prob_test_stage2.csv", "all_test_pred/stage2/prob_test_stage2_model26.csv")
copyfile("model27/prob_test_stage2.csv", "all_test_pred/stage2/prob_test_stage2_model27.csv")
copyfile("model28/prob_test_stage2.csv", "all_test_pred/stage2/prob_test_stage2_model28.csv")
copyfile("model29/prob_test_stage2.csv", "all_test_pred/stage2/prob_test_stage2_model29.csv")


with cd("all_test_pred"):
	print(datetime.now(),"-+- Starting ensemble.py")

	#	IN  :::  problems.csv
	#   OUT :::  problems_flat_train.csv
	#

	system('python ensemble.py')
	#feat_prob.py
	print(datetime.now(),"-+- Done ensemble.py\n")

	print(datetime.now(),"-+- Starting predict.py")


copyfile("all_test_pred/submit_test_stage2_0.74.csv", "final_submission.csv")
























