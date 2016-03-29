"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import numpy as np
import math


# Splits a sequence in n equal parts (it does not preserve the order)
def chunkify(lst, n):
	return [lst[i::n] for i in xrange(n)]


# Creates the training and test sets for the given number of folds
def create_folds(uci, n_folds):
	'''
	@param uci: the dataset
	@type uci: UDataset
	@param n_filds: the number of folds
	@type n_folds: int
	@return: the training and test set for each fold
	@rtype: every fold is a list of tuple (users:int, item:Value)
	'''
	s = {}
	for u in uci.data:
		l = sorted(uci.data[u], key=lambda tup: tup[1].get_time())
		half = int(math.ceil(len(l) / 2.))
		s[u] = (l[:half], l[half:])
		del l
	
	perm = np.random.permutation(uci.num_users())
	chunks = chunkify(perm, n_folds)
	
	for j in xrange(n_folds):
		tr, ts = [], []
		for i, chunk in enumerate(chunks):
			if (i == j):
				for u in chunk:
					for r in s[u][0]:
						tr += [(u,) + r]
						
				for u in chunk:
					for r in s[u][1]:
						ts += [(u,) + r]
			else:
				for u in chunk:
					for c in [0,1]:
						for r in s[u][c]:
							tr += [(u,) + r]
		yield tr, ts
