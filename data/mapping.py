"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

from collections import defaultdict


# Represents the identity mapping for the IDs
class IdentityMapping(object):
	
	# Adds a new explicit ID
	def add(self, explicit):
		return explicit

	# Returns the corresponding explicit ID
	def get_explicit(self, implicit):
		return implicit

	# Returns the corresponding implicit ID
	def get_implicit(self, explicit):
		return explicit


# Represents a natural mapping for the IDs
class Mapping(IdentityMapping):

	def __init__(self):
		self.map_i2e = {}
		self.map_e2i = {}
		self.map_i2e = defaultdict(lambda: -1, self.map_i2e)
		self.map_e2i = defaultdict(lambda: -1, self.map_e2i)
		self.size = 0

	def add(self, explicit):
		i = self.map_e2i[explicit]
		
		if i < 0:
			i = self.size
			self.map_e2i[explicit] = i
			self.map_i2e[i] = explicit
			self.size += 1
		
		return i

	def get_explicit(self, implicit):
		e = self.map_i2e[implicit]
		
		if e != -1: return e
		else: raise IndexError("Index %d does not exists." %implicit)

	def get_implicit(self, explicit):
		i = self.map_e2i[explicit]
		
		if i >= 0: return i
		else: raise IndexError("Index %s does not exists." %explicit)

