"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua

This module contains utility methods for manipulating numpy objects.
"""

from math import sqrt
from sklearn.preprocessing import normalize


# Normalizes the vector
def normalize_vec(v):
	"""
	@param v: the vector
	@type v: cvxopt dense matrix
	@return: the normalized vector
	@rtype: cvxopt dense matrix
	"""
	return v / sqrt(sum(v**2))


# Normalizes the rows of the matrix
def normalize(X):
	"""
	@param X: the matrix
	@type X: numpy dense matrix
	@return: the row-normalized matrix
	@rtype: numpy dense matrix
	"""
	return normalize(X)


# Normalizes the cols of the matrix
def normalize_cols(X):
	"""
	@param X: the matrix
	@type X: numpy dense matrix
	@return: the col-normalized matrix
	@rtype: numpy dense matrix
	"""
	return normalize(X, axis=0)	


# Sorts the vector by value
def sort(v, skip, dec=True):
	"""
	@param v: the vector to sort
	@type v: numpy dense matrix
	@param skip: vector of indexes to skip
	@type skip: numpy dense matrix
	@param dec: whether to sort in descending order or not
	@type dec: boolean (default: True)
	@return: ordered list by value
	@rtype: list
	"""
	d = {k:v[k] for k in xrange(v.size[0]) if k not in skip}
	return sorted(d.keys(),key=lambda s:d[s],reverse=dec)

