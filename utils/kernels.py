"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import scipy.misc as miscs
import utils as ut
from cvxopt.lapack import syev
import numpy as np
import cvxopt as co

def linear(X):
	return X.T * X

# Normalizes the kernel
def normalize_slow(K):
	"""
	@param X: the matrix
	@type X: cvxopt dense matrix
	@return: the row-normalized matrix
	@rtype: cvxopt dense matrix
	"""
	d = ut.cvx.diagonal_vec(K)
	Nr = co.sqrt(d * ut.cvx.ones_vec(K.size[0]).T)
	Nc = co.sqrt(ut.cvx.ones_vec(K.size[0]) * d.T)
	return co.div(co.div(K,Nr), Nc)

def normalize(K):
	YY = ut.cvx.diagonal_vec(K)
	YY = co.sqrt(YY)**(-1)
	return co.mul(K, YY*YY.T)

def spectral_complexity(K):
	return ut.cvx.trace(K) / np.linalg.norm(K, "fro")

def spectral_complexity_norm(K):
	return (spectral_complexity(K) - 1.0) / (np.sqrt(K.size[0]) - 1.0)

@ut.timing
def sparse_polynomial(X, c=1.0, d=2):
	K = X * X.T
	S = co.spmatrix(0.0, [], [], K.size)
	for i in range(1, d+1):
		cmb = miscs.comb(d, i)
		A = co.spmatrix(K.V**i, K.I, K.J)
		S += cmb * c**(d-i) * A
		
	return S

def polynomial(X, b=0.0, d=2.0):
	return ((X * X.T) + b)**d

	
	
	
	
	
	
	
	