"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import cvxopt as co

def KLIN(X):
	return [X*X.T, X.size[1]]

def KXX(KA):
	n = KA.size[0]
	d = co.matrix([KA[i,i] for i in range(n)])
	U = co.matrix([1.0]*n)
	return (d * U.T)

def KNOT(K, n):
	dim = K.size[0]
	Q = co.matrix([[K[i,i]+K[j,j] for i in range(dim)] for j in range(dim)])
	return [n + K - Q, n]
	
def KAND(KA, na, KB, nb):
	n = KA.size[0]
	return [co.mul(KA,KB), na*nb]

def KOR(KA, na, KB, nb):
	return [na*nb \
			- co.mul(KNOT(KXX(KA),na)[0], KNOT(KXX(KB),nb)[0]) \
			- co.mul(KNOT(KXX(KA),na)[0], KNOT(KXX(KB),nb)[0]).T \
			+ co.mul(KNOT(KA,na)[0], KNOT(KB,nb)[0]) \
			, na * nb]

def KXOR(KA, na, KB, nb):
	return [2*KAND(KA,na,KB,nb)[0] \
			+ co.mul(KA,KNOT(KB,nb)[0]) \
			+ co.mul(KNOT(KA,na)[0],KB) \
			- co.mul(KA,KXX(KB).T) \
			+ co.mul(KXX(KA),KXX(KB).T) \
			- co.mul(KXX(KA),KB) \
			- co.mul(KA,KXX(KB)) \
			+ co.mul(KXX(KA).T,KXX(KB)) \
			- co.mul(KXX(KA).T,KB) \
			, na*nb]

def KIMP(KA, na, KB, nb):
	return KOR(KNOT(KA,na)[0],na,KB,nb)
	
def KBIMP(KA, na, KB, nb):
	return KOR(KA,na,KNOT(KB,nb)[0],nb)

def KNIMP(KA, na, KB, nb):
	return KNOT(*(KOR(KNOT(KA,na)[0],na,KB,nb)))

def KEQ(KA,na,KB,nb):
	return KNOT(*(KXOR(KA,na,KB,nb)))

def KNOR(KA,na,KB,nb):
	return KNOT(*(KOR(KA,na,KB,nb)))

def KNAND(KA,na,KB,nb):
	return KNOT(*(KAND(KA,na,KB,nb)))

def KNBIMP(KA, na, KB, nb):
	return KNOT(*(KOR(KA,na,KNOT(KB,nb)[0],nb)))
