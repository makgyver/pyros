"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import sys

from data.reader import CSVReader
import core.engine as exp
import core.evaluation as ev
import data.dataset as ds
from data.mapping import Mapping
import utils as ut
import cvxopt as co
from utils.bool_kernels import *
import numpy as np
from MKLpy.metrics.pairwise import monotone_disjunctive_kernel
import time


def main_simple(argv):
	path_tr, path_ts = argv[0], argv[1]
	separator = " "
	
	# LOAD DATASET
	reader = CSVReader(path_tr, separator)
	uci = ds.UDataset(Mapping(), Mapping())
	reader.read(uci, True)
	#

	# LOAD TESTSET
	rdte = CSVReader(path_ts, separator)
	ucits = ds.UDataset(uci.user_mapping, uci.item_mapping)
	rdte.read(ucits, True)
	#
	
	#################################################
	# Uncomment *only one* of the 'rec' declaration	#
	# to test the corresponding algorithm			#
	#################################################
	
	#rec = exp.I2I_Asym_Cos(uci)
	#rec = exp.CF_OMD(uci)
	#rec = exp.ECF_OMD(uci)
	
	
	# This predictor with the linear kernel should return 
	# almost the same result as ECF_OMD
	#K = ut.kernels.normalize(ut.kernels.md_kernel(uci.to_cvxopt_matrix(),10))
	#print K
	X = np.array(uci.to_cvxopt_matrix())
	#ts = time.time()
	K = [k for _,k in fast_generalized_md_kernel(np.dot(X.T,X),uci.num_users(),4)][-1]
	#te = time.time()
	#print te-ts
	print K
	rec = exp.CF_KOMD(uci, ut.kernels.normalize(co.matrix(K)))
	
	#################################################
	
	print "Training..."
	rec.train(ucits.users)
	
	print "Evaluation..."
	result = ev.evaluate(rec, ucits)
	
	print "Done!"
	print result
	
	
#MAIN
if __name__=='__main__':
	args = sys.argv[1:]
	for i, arg in enumerate(args):
		print (i+1), arg
	
	main_simple(sys.argv[1:])