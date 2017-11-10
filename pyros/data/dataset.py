"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import cvxopt as co
from scipy import sparse
from mapping import IdentityMapping
from reader import Value
import numpy as np


# Abstract RS dataset
class BaseDataset(object):

	def __init__(self,
				user_mapping=IdentityMapping(),
				item_mapping=IdentityMapping()):
		"""
		@param user_mapping: map users id to the integers interval [0, N]
		@type user_mapping: dictionary {any_immutable_type : int}
		@param item_mapping: map items id to the integers interval [0, N]
		@type item_mapping: dictionary {any_immutable_type : int}
		"""
		self.user_mapping = user_mapping
		self.item_mapping = item_mapping
		self.users = set()
		self.items = set()

		self.data = {}

	# Adds a new pair (user, item) to the dataset
	def add(self, user, item):
		"""
		@param user: user id
		@param item: item id
		@return: tuple of new user id and new item id
		@rtype: tuple (int, int)
		"""
		u = self.user_mapping.add(user)
		i = self.item_mapping.add(item)

		self.users.add(u)
		self.items.add(i)

		return u, i

	# Returns the number of users
	def num_users(self):
		"""
		@return: the number of users
		@rtype: int
		"""
		return len(self.users)

	# Returns the number of items
	def num_items(self):
		"""
		@return: the number of items
		@rtype: int
		"""
		return len(self.items)

	# Returns the number of ratings
	def num_ratings(self):
		"""
		@return: the number of users
		@rtype: int
		"""
		raise NotImplementedError

	# Returns the string representation of the dataset
	def __repr__(self):
		"""
		@return: the string representation of the dataset
		@rtype: string
		"""
		return "Data-set with %d users and %d items." %(self.num_users(), self.num_items())

	# Returns the set of items rated by the given user
	def get_items(self, user):
		"""
		@param user: the user
		@type user: int
		@return: set of items rated by user
		@rtype: set of int
		"""
		raise NotImplementedError
	
	# Returns the numpy dense rating matrix corresponding to the dataset
	def to_numpy_matrix(self):
		"""
		@return: the numpy dense rating matrix corresponding to the dataset
		@rtype: numpy dense matrix
		"""
		return NotImplementedError

	# Returns the numpy sparse rating matrix corresponding to the dataset
	def to_numpy_sparse_matrix(self):
		"""
		@return: the numpy dense rating matrix corresponding to the dataset
		@rtype: numpy sparse matrix
		"""
		raise NotImplementedError

	# Returns the cvxopt dense rating matrix corresponding to the dataset
	def to_cvxopt_matrix(self):
		"""
		@return: the cvxopt dense rating matrix corresponding to the dataset
		@rtype: cvxopt dense matrix
		"""
		return NotImplementedError

	# Returns the cvxopt sparse rating matrix corresponding to the dataset
	def to_cvxopt_sparse_matrix(self):
		"""
		@return: the cvxopt sparse rating matrix corresponding to the dataset
		@rtype: cvxopt sparse matrix
		"""
		raise NotImplementedError
	
	# Returns the cvxopt rating matrix corresponding to the dataset
	def to_cvxopt(self, spr=False):
		"""
		@param spr: whether the representation is sparse or not
		@type spr: boolean
		@return: the cvxopt rating matrix corresponding to the dataset
		@rtype: cvxopt matrix
		"""
		return self.to_cvxopt_matrix() if not spr else self.to_cvxopt_sparse_matrix()
	
	# Returns the numpy rating matrix corresponding to the dataset
	def to_numpy(self, spr=False):
		"""
		@param spr: whether the representation is sparse or not
		@type spr: boolean
		@return: the numpy rating matrix corresponding to the dataset
		@rtype: numpy matrix
		"""
		return self.to_numpy_matrix() if not spr else self.to_numpy_sparse_matrix()
	

#Generic RS dataset
class Dataset(BaseDataset):

	def __init__(self,
				user_mapping=IdentityMapping(),
				item_mapping=IdentityMapping()):
		super(Dataset, self).__init__(user_mapping, item_mapping)
		self.scale = Scale()

	def add(self, user, item, rating=Value(0.0)):
		(u, i) = super(Dataset, self).add(user, item)
		self.scale.add(rating.get_float())
		self.data[(u, i)] = rating

		return u, i

	def num_ratings(self):
		return len(self.data)

	def get_items(self, user):
		return set([i for u,i in self.data if u == user])

	def to_numpy_matrix(self):
		result = np.matrix(0.0, (self.num_users(), self.num_items()))
		for (u, i), v in self.data.iteritems():
			result[u, i] = v.get_float()
		return result

	def to_numpy_sparse_matrix(self):
		rows, cols, values = [],[],[]
		for (u, i), v in self.data.iteritems():
			rows += [u]
			cols += [i]
			values += [v.get_float()]

		return sparse.coo_matrix((values, (rows, cols)), shape=(self.num_users(), self.num_items()))

	def to_cvxopt_matrix(self):
		result = co.matrix(0.0, (self.num_users(), self.num_items()))
		for (u, i), v in self.data.iteritems():
			result[u, i] = v.get_float()
		return result
	
	def to_cvxopt_sparse_matrix(self):
		rows, cols, values = [], [], []
		for (u, i), v in self.data.iteritems():
			rows += [u]
			cols += [i]
			values += [v.get_float()]

		return co.spmatrix(values, rows, cols, (self.num_users(), self.num_items()))


#User-centered RS dataset
class UDataset(BaseDataset):

	def __init__(self,
				user_mapping=IdentityMapping(),
				item_mapping=IdentityMapping()):
		super(UDataset, self).__init__(user_mapping, item_mapping)
		self.scale = Scale()
		self.count = 0

	def add(self, user, item, rating=Value(1.0)):
		(u, i) = super(UDataset, self).add(user, item)
		self.scale.add(rating.get_float())
		if u not in self.data:
			self.data[u] = set()
		self.data[u].add((i, rating))
		self.count += 1

		return u, i

	def num_ratings(self):
		return self.count

	def get_items(self, user):
		return set([i for (i,_) in self.data[user]]) if user in self.data else set()

	def to_numpy_matrix(self):
		result = np.zeros((self.num_users(), self.num_items()))
		for u, s in self.data.iteritems():
			for (i, v) in s:
				result[u, i] = v.get_float()
		return result

	def to_numpy_sparse_matrix(self):
		rows, cols, values = [],[],[]
		for u, s in self.data.iteritems():
			for (i, v) in s:
				rows += [u]
				cols += [i]
				values += [v.get_float()]

		return sparse.coo_matrix((values, (rows, cols)), shape=(self.num_users(), self.num_items()))

	def to_cvxopt_matrix(self):
		result = co.matrix(0.0, (self.num_users(), self.num_items()))
		for u, s in self.data.iteritems():
			for (i, v) in s:
				result[u, i] = v.get_float()
		return result
	
	def to_cvxopt_sparse_matrix(self):
		rows, cols, values = [], [], []
		for u, s in self.data.iteritems():
			for (i, v) in s:
				rows += [u]
				cols += [i]
				values += [v.get_float()]
		
		return co.spmatrix(values, rows, cols, (self.num_users(), self.num_items()))


#Item-centered RS dataset
class IDataset(BaseDataset):

	def __init__(self,
				user_mapping=IdentityMapping(),
				item_mapping=IdentityMapping()):
		super(IDataset, self).__init__(user_mapping, item_mapping)
		self.scale = Scale()
		self.count = 0

	def add(self, user, item, rating=Value(1.0)):
		(u, i) = super(IDataset, self).add(user, item)
		self.scale.add(rating.get_float())
		if i not in self.data:
			self.data[i] = set()
		self.data[i].add((u, rating))
		self.count += 1

		return u, i

	def num_ratings(self):
		return self.count
		
	def get_users(self, item):
		return set([u for (u,_) in self.data[item]]) if item in self.data else set()

	def to_numpy_matrix(self):
		result = np.zeros((self.num_users(), self.num_items()))
		for i, s in self.data.iteritems():
			for (u, v) in s:
				result[u, i] = v.get_float()
		return result

	def to_numpy_sparse_matrix(self):
		rows, cols, values = [],[],[]
		for i, s in self.data.iteritems():
			for (u, v) in s:
				rows += [u]
				cols += [i]
				values += [v.get_float()]

		return sparse.coo_matrix((values, (rows, cols)), shape=(self.num_users(), self.num_items()))

	def to_cvxopt_matrix(self):
		result = co.matrix(0.0, (self.num_users(), self.num_items()))
		for i, s in self.data.iteritems():
			for (u, v) in s:
				result[u, i] = v.get_float()
		return result
	
	def to_cvxopt_sparse_matrix(self):
		rows, cols, values = [], [], []
		for i, s in self.data.iteritems():
			for (u, v) in s:
				rows += [u]
				cols += [i]
				values += [v.get_float()]
		
		return co.spmatrix(values, rows, cols, (self.num_users(), self.num_items()))


# Class which represents a scale of values
class Scale(object):

	def __init__(self):
		self.values = set()
		self.min = float("+inf")
		self.max = float("-inf")

	# Adds a value to the scale
	def add(self, value):
		"""
		@param value: the value to add to the scale
		@type value: number
		"""
		self.values.add(value)

		if value > self.max:
			self.max = value
		elif value < self.min:
			self.min = value

#TODO data-set statistics

