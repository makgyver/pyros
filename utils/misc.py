"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import time
import os
import pickle
import cvx as utc
	

def load_dataset(reader):
	uci = reader.read()
	#print uci.num_items()
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


