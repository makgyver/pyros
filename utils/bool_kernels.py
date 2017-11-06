"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import numpy as np
from scipy.special import binom

def fast_generalized_md_kernel(K0, n, D=2):
	N = np.full(K0.shape, n)
	XX = np.dot(np.diag(K0).reshape(K0.shape[0],1), np.ones((1,K0.shape[0])))
	N_x = N - XX
	N_xz = N_x - XX.T + K0
	N_d, N_xd, N_xzd = N.copy(), N_x.copy(), N_xz.copy()
	
	yield 1, N_d - N_xd - N_xd.T + N_xzd
	for d in range(1,D):
		N_d = N_d * (N - d) / (d + 1)
		N_xd = N_xd * (N_x - d) / (d + 1)
		N_xzd = N_xzd * (N_xz - d) / (d + 1)
		yield d+1, N_d - N_xd - N_xd.T + N_xzd
	
	
def fast_generalized_mc_kernel(K0, n, C=2):
	K = K0.copy()	
	yield 1, K
	for c in range(1,C):
		K = K * (K0 - c) / (c + 1)
		yield c+1, K


def fast_generalized_c_kernel(K0, n, C=2):
	N = np.full(K0.shape, n)
	XX = np.dot(np.diag(K0).reshape(K0.shape[0],1), np.ones((1,K0.shape[0])))
	K1 = N - XX - XX.T + 2*K0
	yield 1, K1
	K = K1.copy()
	for c in range(1,C):
		K = K * (K1 - c) / (c + 1)
		yield c+1, K


def fast_generalized_d_kernel(K0, n, D=2):
	c_gen = fast_generalized_c_kernel(K0, n, C=D)
	N = np.full(K0.shape, n)
	Z = np.full(K0.shape, 2)
	T = np.full(K0.shape, 2)
	K = c_gen.next()[1]
	yield 1, K
	for d in range(1,D):
		N = N * (n - d) / (d + 1)
		Z = Z * T
		yield d+1, (Z - T) * N + c_gen.next()[1]


def fast_generalized_mdnf_kernel(K0, n, D=2, C=2):
	for c, cK in fast_generalized_mc_kernel(K0, n, C):	
		for d, dK in fast_generalized_md_kernel(cK, binom(n, c), D):
			yield d, c, dK


def fast_generalized_mcnf_kernel(K0, n, D=2, C=2):
	for d, dK in fast_generalized_md_kernel(K0, n, D):
		for c, cK in fast_generalized_mc_kernel(dK, binom(n, d), C):
			yield d, c, cK

	
def fast_generalized_dnf_kernel(K0, n, D=2, C=2):
	for c, cK in fast_generalized_c_kernel(K0, n, C):
		for d, dK in fast_generalized_md_kernel(cK, 2**c*binom(n, c), D):
			yield d, c, dK


def fast_generalized_cnf_kernel(K0, n, D=2, C=2):
	for d, dK in fast_generalized_d_kernel(K0, n, D):
		for c, cK in fast_generalized_mc_kernel(dK, 2**d*binom(n, d), C):
			yield d, c, cK


def mc_kernel(X, c=1):
	return [k for _,k in fast_generalized_mc_kernel(np.dot(X,X.T), X.shape[1], c)][-1]

def md_kernel(X, d=1):
	return [k for _,k in fast_generalized_md_kernel(np.dot(X,X.T), X.shape[1], d)][-1]
	
def c_kernel(X, c=1):
	return [k for _,k in fast_generalized_c_kernel(np.dot(X,X.T), X.shape[1], c)][-1]

def d_kernel(X, d=1):
	return [k for _,k in fast_generalized_d_kernel(np.dot(X,X.T), X.shape[1], d)][-1]

def mdnf_kernel(X, d=1, c=1):
	return [k for _,_,k in fast_generalized_mdnf_kernel(np.dot(X,X.T), X.shape[1], d, c)][-1]

def mcnf_kernel(X, d=1, c=1):
	return [k for _,_,k in fast_generalized_mcnf_kernel(np.dot(X,X.T), X.shape[1], d, c)][-1]

def cnf_kernel(X, d=1, c=1):
	return [k for _,_,k in fast_generalized_cnf_kernel(np.dot(X,X.T), X.shape[1], d, c)][-1]

def dnf_kernel(X, d=1, c=1):
	return [k for _,_,k in fast_generalized_dnf_kernel(np.dot(X,X.T), X.shape[1], d, c)][-1]
	
