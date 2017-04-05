"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import numpy as np
from sklearn.preprocessing import Binarizer as Skbin, OneHotEncoder, LabelEncoder, MinMaxScaler
from misc import save_as_svmlight
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import sys

'''
Class utility that binarize non-binary datasets
'''
class Binarizer():
	def __init__(self, M, class_index=0):
		self.M = M
		self.class_index = class_index
	
	
	def by_average(self, tune=0.0):
		tune = min(max(-1.0, tune), 1.0) # -1.0 <= tune <= 1.0
		
		A = np.mean(self.M, axis=0)
		R = np.array(self.M, copy=True)
		D = A
		if tune < 0:
			Mn = np.min(self.M, axis=0)
			D = Mn + (1. + tune)*(A - Mn)
		else:
			Mx = np.max(self.M, axis=0)
			D = A + tune*(Mx - A)
		
		for i in range(R.shape[0]):
			for j in range(R.shape[1]):
				R[i,j] = 1.0 if R[i,j] >= D[j] else 0.0
				
		return R
	
	
	def by_percentage(self, perc=.5):
		if perc <= 0. or perc > 1.: #0 <= perc <= 1
			perc = .5
		
		Mn = np.min(self.M, axis=0)
		Mx = np.max(self.M, axis=0)
		D = Mn + (Mx - Mn)*perc
		
		R = np.array(self.M, copy=True)
		for i in range(R.shape[0]):
			for j in range(R.shape[1]):
				R[i,j] = 1.0 if R[i,j] >= D[j] else 0.0
		
		return R
		
		
	def by_threshold(self, threshold=0.0):
		bin = Skbin(threshold).fit(self.M)
		return bin.transform(self.M)
	
	#TODO
	def by_entropy(self):
		pass
	
	
	# SIDE EFFECTS
	def apply_01_scaling(self):
		scaler = MinMaxScaler()
		self.M = scaler.fit_transform(self.M)
		return self
	
	
	# SIDE EFFECTS
	def apply_onehot(self, columns=[]):
		enc = OneHotEncoder()
		enc.fit(self.M[:, columns])
		R = enc.transform(self.M[:, columns]).toarray()
		self.M = np.c_[self.M[:,[x for x in range(self.M.shape[1]) if x not in columns]], R]
		self.class_index -= len([c for c in columns if c < self.class_index])
		return self
		
	
	# SIDE EFFECTS
	def to_numeric(self, columns=[]):
		le = LabelEncoder()
		for i, c in enumerate(columns):
			le.fit(self.M[:, c])
			self.M[:, c] = le.transform(self.M[:, c])
		self.M = self.M.astype(np.float)
		return self
	
	
	# SIDE EFFECTS
	def remove_columns(self, columns=[]):
		self.M = self.M[:, [x for x in range(self.M.shape[1]) if x not in columns or x == self.class_index]]
		self.class_index -= len([c for c in columns if c < self.class_index])
		return self
