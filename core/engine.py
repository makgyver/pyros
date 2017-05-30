"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

from core.recommender import RecEngine
import utils as ut
import utils.cvx as utc
import cvxopt as co
import cvxopt.solvers as solver
import core.baseline as base


'''
Matrix-based implementation of the algorithm described
in "Efficient Top-N Recommendation for Very Large Scale Binary Rated Datasets"
by F. Aiolli
'''
class I2I_Asym_Cos(RecEngine):
	def __init__(self, data, alpha=0.5, q=1.0):
		super(I2I_Asym_Cos, self).__init__(data)
		self.ratings = self.data.to_cvxopt(True)
		self.model = None
		self.alpha = alpha
		self.locality = q

	def train(self, test_users=None):
		self.model = self.ratings.T * self.ratings
		diag = utc.diagonal_vec(self.model)
		row_mat = diag * utc.ones_vec(self.n_items).T
		col_mat = utc.ones_vec(self.n_items) * diag.T

		row_mat **= self.alpha
		col_mat **= 1 - self.alpha
		self.model = co.div(self.model, co.mul(row_mat, col_mat))
		self.model = self.model**self.locality

	def get_scores(self, u):
		return self.model * self.ratings[u, :].T
	
	def get_params(self):
		return {"alpha" : self.alpha, "locality" : self.locality}
	

'''
Implementation of the algorithm described in
"Convex AUC Optimization for Top-N Recommendation with Implicit Feedback"
by F. Aiolli
'''
class CF_OMD(RecEngine):
	def __init__(self, data, lp=1.0, ln=1000.0, spr=False):
		super(CF_OMD, self).__init__(data)
		self.lambda_p = lp
		self.lambda_n = ln
		self.X = utc.normalize_cols(self.data.to_cvxopt(spr))
		self.model = None
		self.sol = {}
		
		
	def train(self, test_users=None):
		self.model = {}
		iset = set(range(self.n_items))
		
		if test_users is None:
			test_users = range(self.n_users)
		
		for i, u in enumerate(test_users):
			#if (i+1) % 100 == 0:
			print "%d/%d" %(i+1, len(test_users))

			Xp_set = self.data.get_items(u)
			Xn_set = iset - Xp_set

			Z = co.spdiag([1.0 if i in Xp_set else -1.0 for i in iset])
			K = 2 * (Z * self.X.T * self.X * Z)
			I = co.spdiag([self.lambda_p if i in Xp_set else self.lambda_n for i in iset])
			P = K + I

			o = utc.zeroes_vec(self.n_items)
			G = -utc.identity(self.n_items)
			A = co.matrix([[1.0 if i in Xp_set else 0.0 for i in iset],
						   [1.0 if j in Xn_set else 0.0 for j in iset]]).T
			b = co.matrix([1.0, 1.0])
			P = co.sparse(P)

			solver.options['show_progress'] = False
			sol = solver.qp(P, o, G, o, A, b)
			self.sol[u] = sol
			self.model[u] = self.X.T * self.X * Z * sol['x']

		# endfor
		return self
	
	def get_params(self):
		return {"lambda_p" : self.lambda_p, "lambda_n" : self.lambda_n}
	
	def get_scores(self, u):
		return self.model[u]


'''
Implementation of the algorithm (which is a simplification of CF-OMD) described in
"Kernel based collaborative filtering for very large scale top-N item recommendation"
by M.Polato and F. Aiolli
'''
class ECF_OMD(RecEngine):
	def __init__(self, data, lp=0.1, spr=False):
		super(ECF_OMD, self).__init__(data)
		self.lambda_p = lp
		self.X = utc.normalize_cols(self.data.to_cvxopt(spr)).T
		self.Xn_ave = utc.ones_vec(self.n_items).T * self.X
		self.model = None
		self.sol = {}
		

	def train(self, test_users=None):
		self.model = {}

		if test_users is None:
			test_users = range(self.n_users)
		
		for i, u in enumerate(test_users):
			if (i+1) % 100 == 0:
				print "%d/%d" %(i+1, len(test_users))

			Xp_list = list(self.data.get_items(u))
			
			np = len(Xp_list)
			nn = float(self.n_items - np)
			
			Xp = co.matrix(self.X[Xp_list, :])
			Xp_ave = utc.ones_vec(np).T * Xp
			Xn_ave_u = (self.Xn_ave - Xp_ave) / nn
			
			kn = Xp * Xn_ave_u.T
			K = Xp * Xp.T
			I = self.lambda_p * utc.identity(np)
			P = K + I
			q = -kn
			G = -utc.identity(np)
			h = utc.zeroes_vec(np)
			A = utc.ones_vec(np).T
			b = co.matrix(1.0)

			solver.options['show_progress'] = False
			sol = solver.qp(P, q, G, h, A, b)
			self.sol[u] = co.matrix(1.0/(self.n_items - np), (self.n_items, 1))
			self.sol[u][Xp_list] = sol['x']
			self.model[u] = self.X * (Xp.T * sol['x'] - Xn_ave_u.T) # not normalized

		# endfor
		return self

	def get_params(self):
		return {"lambda_p" : self.lambda_p}

	def get_scores(self, u):
		return  self.model[u]


'''
Implementation of the algorithm (which is a "kernelification" of ECF-OMD) described in
"Kernel based collaborative filtering for very large scale top-N item recommendation"
by M.Polato and F. Aiolli
'''
class CF_KOMD(RecEngine):
	def __init__(self, data, K=None, lp=0.1, spr=False):
		super(CF_KOMD, self).__init__(data)
		self.lambda_p = lp
		self.K = K
		self.q_ = co.matrix(0.0, (self.n_items, 1))
		for i in xrange(self.n_items):
			self.q_[i,0] = sum(self.K[i,:]) / float(self.n_items) #-1
		
		self.model = None
		self.sol = {} #TODO
		
		
	def train(self, test_users=None):
		self.model = {}
		
		if test_users is None:
			test_users = range(self.n_users)
		
		for i, u in enumerate(test_users):
			#if (i+1) % 100 == 0:
			#	print "%d/%d" %(i+1, len(test_users))

			Xp = list(self.data.get_items(u))
			np = len(Xp)
			
			kp = self.K[Xp, Xp]
			kn = self.q_[Xp,:]
			
			I = self.lambda_p * utc.identity(np)
			P = kp + I
			q = -kn
			G = -utc.identity(np)
			h = utc.zeroes_vec(np)
			A = utc.ones_vec(np).T
			b = co.matrix(1.0)

			solver.options['show_progress'] = False
			sol = solver.qp(P, q, G, h, A, b)
			
			self.model[u] = self.K[Xp,:].T * sol['x'] - self.q_
			#print self.model[u]

		# endfor
		return self

	def get_params(self):
		return {"lambda_p" : self.lambda_p}

	def get_scores(self, u):
		return self.model[u]


class CF_KOMD_CS_POP(CF_KOMD):
	def __init__(self, data, K=None, lp=0.1, spr=False):
		super(CF_KOMD_CS_POP, self).__init__(data, K, lp, spr)
		self.pop_model = base.Popular(data)
	
	@ut.timing
	def train(self, test_users=None):
		self.pop_model.train()
		super(self.__class__, self).train(test_users & self.data.users)
	
	def get_scores(self, u):
		if u in self.model:
			return self.model[u]
		else:
			return self.pop_model.get_scores(u)