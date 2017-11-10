"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

from recommender import RecEngine
import baseline as base
import pyros.utils.cvx as utc
import cvxopt as co
import cvxopt.solvers as solver
import numpy as np
from scipy.sparse import csr_matrix
from pyros.utils.misc import fast_sparse_matrix

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
	
	def train(self, test_users=None):
		self.pop_model.train()
		super(self.__class__, self).train(test_users & self.data.users)
	
	def get_scores(self, u):
		if u in self.model:
			return self.model[u]
		else:
			return self.pop_model.get_scores(u)

	

'''
Implementation of the algorithm SLIM described in
"SLIM: Sparse Linear Methods for Top-N Recommender Systems"
by Xia Ning and George Karypis
'''
class SLIM(RecEngine):
	
	def __init__(self, data, beta=4, lam=.5):
		super(self.__class__, self).__init__(data)
		self.R = self.data.to_cvxopt(False)
		self.beta = beta
		self._lambda = lam
		self.model = None
	
	'''
	Load the model from a .mat file, which is the type of output given by the
	SLIM implementation provided by the authors (https://www-users.cs.umn.edu/~ningx005/slim/html/)
	'''
	def load_model(self, filename):
		f = open(filename, "rb")
		self.model = {}
		content = f.readlines()
		for u, line in enumerate(content):
			#if (u+1) % 500 == 0:
			#	print "%d/%d" %(u+1, len(content))
			uss = line.strip().split(" ")
			ss = co.matrix(0.,(1,self.n_users))
			if len(uss) > 1:
				for x in range(0,len(uss),2):
					ss[0,int(uss[x][0])-1] = float(uss[x+1])
			self.model[u] = self.R.T * ss.T
	
	def train(self, test_users=None):
		self.model = {}
		
		if test_users is None:
			test_users = range(self.n_users)
		
		RR = self.R*self.R.T 
		P = RR + self.beta * utc.identity(self.n_users) 
		G = -utc.identity(self.n_users)
		h = utc.zeroes_vec(self.n_users)
		b = utc.zeroes_vec(1)		
		
		for i, u in enumerate(test_users):
			#if (i+1) % 100 == 0:
			#print "%d/%d" %(i+1, len(test_users))
			q = self._lambda * utc.ones_vec(self.n_users) - RR[u,:].T 
			A = co.matrix(0.,(1,self.n_users))
			A[0,u] = 1.

			solver.options['show_progress'] = False
			sol = solver.qp(P, q, G, h, A, b)
			
			self.model[u] = self.R.T * sol['x']
		
		return self
		

	def get_params(self):
		return {"lambda" : self._lambda, "beta" : self.beta}

	def get_scores(self, u):
		return  self.model[u]


'''
Implementation of the algorithm WRMF described in
"Collaborative Filtering for Implicit Feedback Datasets"
by Yifan Hu, Yehuda Koren and Chris Volinsky
and in 
"One-Class Collaborative Filtering"
by Rong Pan, Yunhong Zhou, Bin Cao, Nathan N. Liu, Rajan Lukose, Martin Scholz and Qiang Yang
'''
class WRMF(RecEngine):
	'''
	latent_factors : int
		Number of latent factors.
	alpha : float
		Confidence weight, confidence c = 1 + alpha*r where r is the observed "rating".
	lbda : float
		Regularization constant.
	num_iters : int
		Number of iterations of alternating least squares.
	'''
	def __init__(self, data, latent_factors, alpha=0.01, lbda=0.015, num_iters=15):
		super(self.__class__, self).__init__(data)
		self.R = fast_sparse_matrix(self.data.to_numpy_sparse_matrix().tocsr())
		self.latent_factors = latent_factors
		self.alpha = alpha
		self.lbda = lbda
		self.num_iters = num_iters
		self.model = None
		
	def init_factors(self,num_factors,assign_values=True):
		if assign_values:
			return self.latent_factors**-0.5*np.random.random_sample((num_factors,self.latent_factors))
		return np.empty((num_factors,self.latent_factors))

	def train(self, test_users=None):
		num_users,num_items = self.R.shape

		self.U = self.init_factors(num_users, False)  # don't need values, will compute them
		self.V = self.init_factors(num_items)
		for it in xrange(self.num_iters):
			print 'iteration',it
			# fit user factors
			VV = self.V.T.dot(self.V)
			for u in xrange(num_users):
				# get (positive i.e. non-zero scored) items for user
				indices = self.R.X[u].nonzero()[1]
				if indices.size:
					self.U[u,:] = self.update(indices,self.V,VV)
				else:
					self.U[u,:] = np.zeros(self.latent_factors)
			# fit item factors
			UU = self.U.T.dot(self.U)
			for i in xrange(num_items):
				indices = self.R.fast_get_col(i).nonzero()[0]
				if indices.size:
					self.V[i,:] = self.update(indices,self.U,UU)
				else:
					self.V[i,:] = np.zeros(self.latent_factors)
		
		self.model = np.dot(self.U,self.V.T)
		return self

	def update(self,indices,H,HH):
		Hix = H[indices,:]
		M = HH + self.alpha*Hix.T.dot(Hix) + np.diag(self.lbda*np.ones(self.latent_factors))
		return np.dot(np.linalg.inv(M),(1+self.alpha)*Hix.sum(axis=0))
	
	def get_params(self):
		return {'latent_factors':self.latent_factors, 'alpha':self.alpha, 'lambda':self.lbda, 'num_iters':self.num_iters}
	
	def get_scores(self, u):
		return co.matrix(self.model[u])
	

'''
Implementation of the algorithm BPRMF described in
"BPR: Bayesian Personalized Ranking from Implicit Feedback"
by Steffen Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme
'''
class BPRMF(RecEngine):
	def __init__(self, data, factors=40, learn_rate=0.05, num_iters=15, 
				 init_mean=0.1, init_stdev=0.1, reg_u=0.0025, reg_i=0.0025, reg_bias=0):
		super(self.__class__, self).__init__(data)

		self.factors = factors
		self.learn_rate = learn_rate
		self.init_mean = init_mean
		self.init_stdev = init_stdev
		self.num_iters = num_iters 
		self.reg_bias = reg_bias
		self.reg_u = reg_u
		self.reg_i = reg_i
		self.reg_j = reg_i
		
		# internal vars
		self.loss = None
		self.loss_sample = list()
		self._create_factors()

	def _create_factors(self):
		self.p = np.random.normal(self.init_mean, self.init_stdev, (self.data.num_users(), self.factors))
		self.q = np.random.normal(self.init_mean, self.init_stdev, (self.data.num_items(), self.factors))
		# self.bias = self.init_mean * np.random.randn(self.number_items, 1) + self.init_stdev ** 2
		self.bias = np.zeros(self.data.num_items(), np.double)

	def _sample_triple(self):
		user = np.random.choice(list(self.data.users))
		user_items = list(self.data.get_items(user))
		item_i = np.random.choice(user_items)
		item_j = np.random.choice(list(self.data.items - self.data.get_items(user)))
		return user, item_i, item_j

	def _predict(self, user, item):
		return self.bias[item] + np.dot(self.p[user], self.q[item])

	def _update_factors(self, user, item_i, item_j):
		# Compute Difference
		eps = 1 / (1 + np.exp(self._predict(user, item_i) - self._predict(user, item_j)))

		self.bias[item_i] += self.learn_rate * (eps - self.reg_bias * self.bias[item_i])
		self.bias[item_j] += self.learn_rate * (eps - self.reg_bias * self.bias[item_j])

		# Adjust the factors
		u_f = self.p[user]
		i_f = self.q[item_i]
		j_f = self.q[item_j]

		# Compute and apply factor updates
		self.p[user] += self.learn_rate * ((i_f - j_f) * eps - self.reg_u * u_f)
		self.q[item_i] += self.learn_rate * (u_f * eps - self.reg_i * i_f)
		self.q[item_j] += self.learn_rate * (-u_f * eps - self.reg_j * j_f)

	def _compute_loss(self):
		ranking_loss = 0
		for sample in self.loss_sample:
			x_uij = self._predict(sample[0], sample[1]) - self._predict(sample[0], sample[2])
			ranking_loss += 1 / (1 + np.exp(x_uij))

		complexity = 0
		for sample in self.loss_sample:
			complexity += self.reg_u * np.linalg.norm(self.p[sample[0]]) ** 2
			complexity += self.reg_i * np.linalg.norm(self.q[sample[1]]) ** 2
			complexity += self.reg_j * np.linalg.norm(self.q[sample[2]]) ** 2
			complexity += self.reg_bias * self.bias[sample[1]] ** 2
			complexity += self.reg_bias * self.bias[sample[2]] ** 2

		return ranking_loss + 0.5 * complexity

	# Perform one iteration of stochastic gradient ascent over the training data
	# One iteration is samples number of positive entries in the training matrix times
	def train(self, test_users=None):
		num_sample_triples = int(np.sqrt(len(self.data.users)) * 100)
		for _ in range(num_sample_triples):
			self.loss_sample.append(self._sample_triple())
		self.loss = self._compute_loss()

		for n in range(self.num_iters):
			for _ in range(self.data.num_ratings()):
				u, i, j = self._sample_triple()
				self._update_factors(u, i, j)

			actual_loss = self._compute_loss()
			if actual_loss > self.loss:
				self.learn_rate *= 0.5
			elif actual_loss < self.loss:
				self.learn_rate *= 1.1
			self.loss = actual_loss

			print "step::", n, "RMSE::", self.loss, "lrate::", self.learn_rate
		return self

	def get_params(self):
		return {"factors": self.factors, "learn_rate": self.learn_rate, "": self.num_iters, "reg_user" : self.reg_u, "reg_item": self.reg_i, "reg_bias" : self.reg_bias}

	def get_scores(self, u):
		model_u = np.dot(self.p[[u]], self.q.T) + self.bias
		return co.matrix(model_u.T)

