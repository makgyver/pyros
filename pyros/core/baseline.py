"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import cvxopt as co
import pyros.utils.cvx as ut
from recommender import RecEngine


class Random(RecEngine):
	def __init__(self, data):
		super(self.__class__, self).__init__(data)
		self.rand = None

	def train(self):
		self.rand = co.uniform(self.n_users, self.n_items)
		return self

	def get_scores(self, u):
		return self.rand[u, :]


class Popular(RecEngine):
	def __init__(self, data):
		super(self.__class__, self).__init__(data)
		self.popular = None

	def train(self):
		ratings = self.data.to_cvxopt_matrix()
		self.popular = (ut.ones_vec(self.n_users).T * ratings).T
		return self

	def get_scores(self, u):
		return self.popular


class Constant(RecEngine):
	def __init__(self, data, k=None):
		super(self.__class__, self).__init__(data)
		if k is None:
			self.constant = (self.data.scale.max - self.data.scale.min) / 2.0
		else:
			self.constant = k
	
	def get_params(self):
		return {"constant" : self.constant}
	
	def get_scores(self, u):
		return co.matrix(self.constant, (self.n_items,1))


