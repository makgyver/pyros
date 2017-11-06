"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua

This module contains classes to read data sets from files.
By default, files are considered CSV (Comma Separated Values) with
a space as separator. Each line of the files are meant to be a record
with this form:

user_id || item_id || v :subscript:'1' || v :subscript:'2' || ... || v :subscript:'n',

where v :subscript:'i' are some kind of values and || are separators (spaces by default).
"""


class CSVReader(object):

	def __init__(self, filename, delimiter=' ', binarize=False):
		"""
		@param filename: the filename of the CSV file
		@type filename: string
		@param delimiter: the delimiter of the fields
		@type delimiter: string
		"""
		self.filename = filename
		self.delimiter = delimiter
		self.binarize = binarize

	# Reads the file and create the data set
	def read(self, data, implicit=False):
		"""
		@param data: the empty dataset
		@type data: BaseDataset
		@return: a dataset
		@rtype: UDataset
		"""
		with open(self.filename, 'r+') as f:
			contents = f.read().split("\n")
		
		for line in contents:
			r = line.strip().split(self.delimiter)
			if len(r) >= 2:
				(u, i, v) = self.interpret(r)
				if (u != -1):
					if implicit:
						data.add(u, i)
					else:
						data.add(u, i, v)
						


	# Interprets a single splitted line
	def interpret(self, args):
		"""
		@param args: strings' tuple
		@type args: tuple of strings 
		"""
		if len(args) == 2:
			return (int(args[0]), int(args[1]), Value(1.0))
		elif len(args) > 2:
			return (int(args[0]), int(args[1]), Value(1.0) if self.binarize else Value(float(args[2])))
		else: #ERROR
			return (-1, -1, None)


class MLReader(CSVReader):
	
	def read(self, data):
		with open(self.filename, 'r+') as f:
			contents = f.read().split("\n")
		
		for line in contents:
			r = line.strip().split(self.delimiter)
			if len(r) >= 2:
				(u, i, v) = self.interpret(r)
				if u != -1:
					data.add(u, i, v)

	
	def __init__(self, filename, delimiter=" "):
		super(MLReader, self).__init__(filename, delimiter)
	
	def interpret(self, args):
		if (len(args) > 3):
			v = Value(1.0, int(args[3])) if self.binarize else Value(float(args[2]), int(args[3]))
			return (int(args[0]), int(args[1]), v)
		else:
			return (int(args[0]), int(args[1]), Value(1.0))


class MSDReader(CSVReader):
	
	def __init__(self, filename, delimiter="\t"):
		super(MSDReader, self).__init__(filename, delimiter)
	
	def interpret(self, args):
		return (args[0], args[1], Value(1.0))


# Represents a generic value for the ratings
class Value(object):
	
	def __init__(self, v, t=0):
		"""
		@param v: a generic value
		@type v: float
		@param t: the timestamp (default: 0)
		@type t: int 
		"""
		self.val = v
		self.time = t
	
	# Returns the value
	def get_float(self):
		return self.val
	
	# Returns the timestamp
	def get_time(self):
		return self.time

	def __eq__(self, obj):
		return self.__dict__ == obj.__dict__
	
	def __hash__(self):
		return hash((self.val, self.time))

	def __repr__(self):
		if self.time != 0:
			return "(%s, %s)" %(self.val, self.time)
		else:
			return "%s" %self.val 
