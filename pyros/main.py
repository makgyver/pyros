"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import sys

from pyros.data.reader import CSVReader
import pyros.core.engine as exp
import pyros.core.evaluation as ev
import pyros.data.dataset as ds
from pyros.data.mapping import Mapping
import pyros.utils as ut
import cvxopt as co
from pyros.utils.bool_kernels import *
import numpy as np


def main_simple(argv):
	path_tr, path_ts = argv[0], argv[1]
	separator = " "
	
	# LOAD DATASET
	reader = CSVReader(path_tr, separator)
	train_set = ds.UDataset(Mapping(), Mapping())
	reader.read(train_set, True)
	#

	# LOAD TESTSET
	rdte = CSVReader(path_ts, separator)
	test_set = ds.UDataset(train_set.user_mapping, train_set.item_mapping)
	rdte.read(test_set, True)
	#
	
	#################################################
	# Uncomment *only one* of the 'rec' declaration	#
	# to test the corresponding algorithm			#
	#################################################
	
	#rec = exp.I2I_Asym_Cos(train_set)
	#rec = exp.CF_OMD(train_set)
	#rec = exp.ECF_OMD(train_set)
	#rec = exp.WRMF(train_set)
	#rec = exp.SLIM(train_set)
	#rec = exp.BPRMF(train_set)
	
	X = np.array(train_set.to_cvxopt_matrix())
	d = 4
	K = mc_kernel(X.T, d)
	rec = exp.CF_KOMD(train_set, ut.kernels.normalize(co.matrix(K)))
	
	#################################################
	
	print "Training..."
	rec.train(test_set.users)
	
	print "Evaluation..."
	result = ev.evaluate(rec, test_set)
	
	print "Done!"
	print result
	
	
#MAIN
if __name__=='__main__':
	args = sys.argv[1:]
	for i, arg in enumerate(args):
		print (i+1), arg
	
	main_simple(sys.argv[1:])