"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import scipy.misc as miscs
from scipy.special import binom
import utils as ut
from cvxopt.lapack import syev
import numpy as np
import cvxopt as co
import math
import mpmath as mp


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
	"""
	@param K: the matrix
	@type K: cvxopt dense matrix
	@return: the row-normalized matrix
	@rtype: cvxopt dense matrix
	"""
	YY = ut.cvx.diagonal_vec(K)
	YY = co.sqrt(YY)**(-1)
	return co.mul(K, YY*YY.T)


def spectral_complexity(K):
	a = ut.cvx.trace(K)
	b = np.linalg.norm(K, "fro")
	#print a, b
	#return ut.cvx.trace(K) / np.linalg.norm(K, "fro")
	return a / b


def spectral_complexity_norm(K):
	return (spectral_complexity(K) - 1.0) / (np.sqrt(K.size[0]) - 1.0)


def tanimoto(X):
	d = co.matrix([sum(X[:,i].V) for i in xrange(X.size[1])])
	Yp = ut.ones_vec(X.size[1]) * d.T
	Xp = d * ut.ones_vec(X.size[1]).T
	Kl = X.T * X
	K = co.div(Kl, Xp + Yp - Kl)
	#K = ut.kernels.normalize(K)
	return K


def sparse_polynomial(X, c=1.0, d=2):
	K = X.T * X
	S = co.spmatrix(0.0, [], [], K.size)
	for i in range(1, d+1):
		cmb = miscs.comb(d, i)
		A = co.spmatrix(K.V**i, K.I, K.J)
		S += cmb * c**(d-i) * A
		
	return S
	

def polynomial(X, b=0.0, d=2.0):
	return ((X * X.T) + b)**d


@ut.timing
def superset_kernel(R, k, norm=True):
	
	n = R.size[0]
	m = R.size[1]
	
	x_choose_k = [0]*(n+1)
	x_choose_k[0] = 0
	for i in range(1, n+1):
		x_choose_k[i] = binom(i,k)
	
	nCk = x_choose_k[n]
	X = R.T*R
	
	K = co.matrix(0.0, (X.size[0], X.size[1]))
	#print K.size
	#K = co.spmatrix(0.0, [], [], X.size)
	for i in range(m):
		#if (i+1) % 100 == 0:
		#	print "%d/%d" %(i+1,m)
		for j in range(i, m):
			n_niCk = x_choose_k[n-int(X[i,i])]
			n_njCk = x_choose_k[n-int(X[j,j])]
			n_ni_nj_nijCk = x_choose_k[n-int(X[i,i])-int(X[j,j])+int(X[i,j])]
			#print n_niCk, n_njCk, n_ni_nj_nijCk
			K[i,j] = K[j,i] = nCk - n_niCk - n_njCk + n_ni_nj_nijCk
	
	#print ut.diagonal_vec(K)
	if norm:
		K = ut.kernels.normalize(K)		
	return K

@ut.timing
def anova_bin_kernel(R, k, norm=True):
	
	n = R.size[0]
	m = R.size[1]
	X = R.T*R

	x_choose_k = [0]*(n+1)
	x_choose_k[0] = 0
	for i in range(1, n+1):
		x_choose_k[i] = binom(i,k)

	K = co.matrix(0.0, (m,m))
	for i in range(m):
		for j in range(i, m):
			K[i,j] = K[j,i] = x_choose_k[int(X[i,j])]
			
	if norm:
		for i in range(m):
			if K[i,i] == 0:
				K[i,i] = 1.0
				
		K = ut.kernels.normalize(K)		
	return K

@ut.timing
def mdnf_kernel(R, norm=True):
	
	n = R.size[0]
	m = R.size[1]
	X = R.T*R

	K = co.matrix(0.0, (m,m))
	for i in range(m):
		for j in range(i, m):
			K[i,j] = K[j,i] = 2.0**int(X[i,j]) - 1.0
			
	if norm:
		K = ut.kernels.normalize(K)		
	return K


@ut.timing
def mdnf_kernel_plus(R):
	
	n = R.size[0]
	m = R.size[1]
	X = R.T*R

	
	d = {}
	for i in range(m):
		d[i] = mp.mpf(2**int(X[i,i]) - 1)
	
	K = co.matrix(0.0, (m,m))
	for i in range(m):
		if (i+1) % 100 == 0:
			print "%d/%d" %(i+1,m)
		for j in range(i+1, m):
			K[i,j] = K[j,i] = float(mp.mpf(2**int(X[i,j]) - 1) / (mp.sqrt(d[i])*mp.sqrt(d[j])))
			
	for i in range(m):
		K[i,i] = 1.0	
		
	return K

class Binomemoize():
	def __init__(self):
		self.mem = {}
	
	def go(self, n, k):
		if n in self.mem:
			if k not in self.mem[n]:
				self.mem[n][k] = binom(n, k)
		else:
			self.mem[n] = {}
			self.mem[n][k] = binom(n, k)
		
		return self.mem[n][k]

def k_over_d(k, d, bin, n, nij, ni_nij, nj_nij, n_ni_nj_nij):
	
	if k == 0:
		return bin.go(n, d)

	A1 = 0
	for s in range(k):
		A1 += bin.go(nij, s) * bin.go(ni_nij, k-s-1) * bin.go(nj_nij, k-s-1) * bin.go(n_ni_nj_nij, d+s-2*(k-1))
		  
	A2 = 0
	for s in range(k):
		D = 0
		for t in range(k-s, d-k+2):
			D += bin.go(ni_nij, t) * bin.go(n_ni_nj_nij, d-t-k+1)
		A2 += bin.go(nij, s) * bin.go(nj_nij, k-s-1) * D

	A3 = 0        
	for s in range(k):
		D = 0
		for t in range(k-s, d-k+2):
			D += bin.go(nj_nij, t) * bin.go(n_ni_nj_nij, d-t-k+1)
		A3 += bin.go(nij, s) * bin.go(ni_nij, k-s-1) * D

	return k_over_d(k-1, d, bin, n, nij, ni_nij, nj_nij, n_ni_nj_nij) - A1 - A2 - A3

@ut.timing
def kd_kernel(R, d, k, norm=True):

	n = R.size[0]
	m = R.size[1]
	X = R.T*R
	
	bin = Binomemoize()
	
	K = co.matrix(0.0, (m,m))
	for i in range(m):
		#if (i+1) % 100 == 0:
		#	print "%d/%d" %(i+1,m)
		for j in range(i, m):
			K[i,j] = K[j,i] = k_over_d(k, d, bin, n, int(X[i,j]), int(X[i,i])-int(X[i,j]), int(X[j,j])-int(X[i,j]), n-int(X[i,i])-int(X[j,j])+int(X[i,j]))
			
	if norm:
		for i in range(m):
			if K[i,i] == 0:
				K[i,i] = 1.0
				
		K = ut.kernels.normalize(K)		
	return K
	
	