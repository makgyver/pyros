"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import time
import os
import pickle
import cvx as utc
import numpy as np
from sklearn.datasets import dump_svmlight_file


def load_dataset(reader):
	uci = reader.read()
	R = uci.to_cvxopt_sparse_matrix()
	X = utc.normalize_cols_sparse(R)
	
	return X, R, uci


# Saves the object to file
def save_to_file(obj, fname):
	"""
	@param obj: the object to save
	@param fname: the file name
	@type fname: string
	"""
	pickle.dump(obj, open(fname, "wb"))


# Loads object from file
def load_from_file(fname):
	"""
	@param fname: the file name
	@type fname: string
	@return: the object
	"""
	return pickle.load(open(fname, "rb"))


# Saves results to file
def save_results(path, rec, result):
	with open(path, "w") as f:
		f.write(rec.get_fullname() + "\n")
		f.write(result)


# Sorts the list by value
def sort(l, skip, dec=True):
	"""
	@param l: the list to sort
	@type l: list of numbers
	@param skip: list of indexes to skip
	@type skip: list of int
	@param dec: whether to sort in descending order or not
	@type dec: boolean (default: True)
	"""
	d = {k:l[k] for k in xrange(len(l)) if k not in skip}
	return sorted(d.keys(),key=lambda s:d[s],reverse=dec)
	
	
# Calculates the time taken by the given method/function
def timing(method):
	"""
	@param method: the method to time
	@type method: function closure
	"""
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()

		#print '%r (%r, %r) %2.2f sec' % \
		#(method.__name__, args, kw, te-ts)
		print '%r %r %2.2f sec' %(method.__name__, args, te-ts)
		#logging.debug('%r %r %2.2f sec' %(method.__name__, args, te-ts))
		return result
	return timed

def save_as_svmlight(M, yid, fname):
	dump_svmlight_file(M[:,[x for x in range(M.shape[1]) if x != yid]], M[:,yid], fname)


class Binomemoize():
	def __init__(self):
		self.mem = {}
	
	def go(self, n, k):
		if n in self.mem:
			if k not in self.mem[n]:
				self.mem[n][k] = binom(n, k)
		else:
			self.mem[n] = {k : binom(n, k)}
		
		return self.mem[n][k]

def bignom(n,k):
	if n < k: return mp.mpf(0.0)
	res = mp.mpf(1.0)
	for i in range(0,k):
		res = res * mp.mpf(float(n-i)/(k-i))
	return res


class fast_sparse_matrix():
	def __init__(self,X,col_view=None):
		self.X = X.tocsr()
		if col_view is not None:
			self.col_view = col_view
		else:
			# create the columnar index matrix
			ind = self.X.copy()
			ind.data = np.arange(self.X.nnz)
			self.col_view = ind.tocsc()

	@property
	def shape(self):
		return self.X.shape

	def fast_get_col(self,j):
		col = self.col_view[:,j].copy()
		col.data = self.X.data[col.data]
		return col
