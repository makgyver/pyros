"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import numpy as np

def scaled_md_kernel(K0, n, D=2):
	N = np.full(K0.shape, n)
	KXX = np.dot(np.diag(K0).reshape(K0.shape[0],1), np.ones((1,K0.shape[0])))
	XX = np.full(K0.shape, 1.)
	ZZ = XX.copy()
	XZ = XX.copy()

	for d in range(1,D+1):
		XX = np.multiply(XX, np.divide(N-KXX-d+1, N-d+1))
		ZZ = np.multiply(ZZ, np.divide(N-KXX.T-d+1, N-d+1))
		XZ = np.multiply(XZ, np.divide(N-KXX-KXX.T+K0-d+1, N-d+1))
		yield d, 1. - XX - ZZ + XZ

def scaled_mc_kernel(K0, n, C=2):
	N = np.full(K0.shape, n)
	XZ = np.full(K0.shape, 1.)

	for c in range(1,C+1):
		XZ = np.multiply(XZ, np.divide(K0-c+1, N-c+1.))
		yield c, XZ

def scaled_md(X, d):
	K0 = np.dot(X, X.T) 
	return [k for _,k in scaled_md_kernel(K0, X.shape[1], d)][-1]

def scaled_mc(X, c):
	K0 = np.dot(X, X.T) 
	return [k for _,k in scaled_mc_kernel(K0, X.shape[1], c)][-1]
