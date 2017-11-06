"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua

This module contains utility methods for manipulating cvxopt objects.
"""

import cvxopt as co
from math import sqrt


# Creates a n-dimensional vector initialized with all zeroes
def zeroes_vec(n):
	"""
	@param n: the dimension of the vector
	@type n: int
	@return: a column n-dimensional zero vector 
	@rtype: cvxopt dense matrix
	"""
	return co.matrix(0.0, (n,1))


# Creates a n-dimensional vector initialized with all ones
def ones_vec(n):
	"""
	@param n: the dimension of the vector
	@type n: int
	@return: a column n-dimensional one vector 
	@rtype: cvxopt dense matrix
	"""
	return co.matrix(1.0, (n,1))


# Returns the diagonal of the given matrix
def diagonal_vec(X):
	"""
	@param X: dense square matrix
	@type X: cvxopt dense matrix
	@return: a column vector with the diagonal elements of X
	@rtype: cvxopt dense matrix
	"""
	return co.matrix([X[i,i] for i in xrange(X.size[0])])


# Creates a diagonal matrix using the given vector as diagonal
def diag(v):
	"""
	@param v: the vector to put in the diagonal
	@type v: cvxopt dense matrix
	@return: diagonal matrix
	@rtype: cvxopt dense matrix
	"""
	result = co.matrix(0.0, (v.size[0], v.size[0]))
	for i, x in enumerate(v):
		result[i, i] = x
	return result


# Creates the identity matrix of the given dimension
def identity(n):
	"""
	@param n: the dimension of the matrix (n by n)
	@type n: int
	@return: the identity matrix
	@rtype: cvxopt dense matrix
	"""
	return diag(ones_vec(n))


# Calculates the trace of the matrix
def trace(K):
	"""
	@param K: square matrix
	@type K: cvxopt dense matrix
	@return: the trace of the matrix
	@rtype: float
	"""
	return sum(diagonal_vec(K))


# Normalizes the vector
def normalize_vec(v):
	"""
	@param v: the vector
	@type v: cvxopt dense matrix
	@return: the normalized vector
	@rtype: cvxopt dense matrix
	"""
	return v / sqrt(sum(v**2))


# Divides the rows for the sum of its elements
def normalize_rows(X):
	"""
	@param X: the matrix
	@type X: cvxopt dense matrix
	@return: the row-normalized matrix
	@rtype: cvxopt dense matrix
	"""
	d = diagonal_vec(X*X.T)
	N = co.sqrt(d * ones_vec(X.size[1]).T)
	return co.div(X,N)


# Divides the rows for the sum of its elements
def normalize_cols(X):
	"""
	@param X: the matrix
	@type X: cvxopt dense matrix
	@return: the col-normalized matrix
	@rtype: cvxopt dense matrix
	"""
	d = diagonal_vec(X.T*X)
	N = co.sqrt(ones_vec(X.size[0])*d.T)
	return co.div(X,N)

# Divides the columns for the sum of its elements
def normalize_cols_sparse(X):
	"""
	@param X: the matrix
	@type X: cvxopt matrix
	@return: the col-normalized matrix
	@rtype: cvxopt sparse matrix
	"""
	d = co.matrix([sum(X[:,i].V) for i in xrange(X.size[1])])
	N = co.sqrt(d.T)
	I, J = X.I, X.J
	V = []
	for j in X.J:
		V += [(1.0 / N[j]) if N[j] != 0.0 else 0.0]
	return co.spmatrix(V, I, J)

#def min_enclosing_ball(X):
	

# Sorts the vector by value
def sort(v, skip, dec=True):
	"""
	@param v: the vector to sort
	@type v: cvxopt dense matrix
	@param skip: vector of indexes to skip
	@type skip: cvxopt dense matrix
	@param dec: whether to sort in descending order or not
	@type dec: boolean (default: True)
	@return: ordered list by value
	@rtype: list
	"""
	d = {k:v[k] for k in xrange(v.size[0]) if k not in skip}
	return sorted(d.keys(),key=lambda s:d[s],reverse=dec)


# Applies the sigmoid function to the matrix
def sigmoid(X):
	return (1.0 + co.exp(-X))**(-1)

# Calculates the density of the matrix
def density(X):
	z = 0
	for i in range(X.size[0]):
		for j in range(X.size[1]):
			z += X[i,j] > 0.0
	
	return float(z) / (X.size[0] * X.size[1])