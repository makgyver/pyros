"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import sys

from data.reader import CSVReader
import core.evaluation as ev
import core.baseline as base
import data.dataset as ds
from data.mapping import Mapping
import utils as ut
import cvxopt as co


def news_rec(args):
	
	day = int(args[0])
	path = args[1]
	
	print "DAY: %d\n" %day
	
	hours = [chr(i) for i in range(97, 97+24)]
	train_set = ds.UDataset(Mapping(), Mapping())
	
	for i in range(len(hours)-1):
		print "HOUR: %d" %i
		
		#TRAINING SET		
		for j in range(i+1):
			reader = CSVReader("%s/data_day_%d_csv/xa%c_%d.data" %(path, day, hours[j], day), " ")
			reader.read(train_set, True)
		
		print "TRAIN USERS: %d" %train_set.num_users()
		print "TRAIN ITEMS: %d" %train_set.num_items()
		
		#TEST SET
		rdte = CSVReader("%s/data_day_%d_csv/xa%c_%d.data" %(path, day, hours[i+1], day), " ")
		test_set = ds.UDataset(train_set.user_mapping, train_set.item_mapping)
		rdte.read(test_set, True)
		
		print "TEST USERS: %d" %test_set.num_users()
		print "TEST ITEMS: %d" %test_set.num_items()
		print "USERS INTERSECTION: %d" %(len(train_set.users & test_set.users))
		print "ITEMS INTERSECTION: %d" %(len(train_set.items & test_set.items)) 
			
		#TRAINING PHASE
		
		rec = base.Popular(train_set)
		
		print "Training..."
		rec.train()
		
		#EVALUATION PHASE
		
		print "Evaluation..."
		result = ev.evaluate(rec, test_set)
		
		
		print "Done!"
		print result
		print
	
	
#MAIN
if __name__=='__main__':
	args = sys.argv[1:]
	for i, arg in enumerate(args):
		print (i+1), arg
	
	news_rec(sys.argv[1:])