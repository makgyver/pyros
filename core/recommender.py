"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

# Base class for recommendation engine
class RecEngine(object):

	def __init__(self, data):
		"""
		@param data: the dataset
		@type data: Dataset
		"""
		self.data = data
		self.n_users = data.num_users()
		self.n_items = data.num_items()

	# Trains the model
	def train(self, test_users=None):
		"""
		@return: the model itself
		@rtype: RecEngine
		"""
		return self
	
	# Returns the items' score for the user
	def get_scores(self, u):
		raise NotImplementedError

	# Gets the method's parameters
	def get_params(self):
		return {}
	
	# Sets the method's parameters
	def set_params(self, **params):
		"""
		@param params: parameters of the method
		@type params: dictionary {string: any_type}
		@return: the model itself
		@rtype: RecEngine
		"""
		for parameter, value in params.items():
			self.setattr(parameter, value)
		return self
	
	# Returns the name of the class with its parameters
	def get_fullname(self):
		return "%s : %s" %(self.__class__.__name__, self.get_params())

