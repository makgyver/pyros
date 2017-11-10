"""
@author: Mirko Polato
@contact: mpolato@math.unipd.it
@organization: University of Padua
"""

import numpy as np
import math
from pyros.utils import misc


# Calculates the AUC
def auc(ord_pred, target):
	"""
	@param ord_pred: ordere predicted scores
	@type ord_pred: iterable sequence of int
	@param target: set of positive items
	@type target: set of int
	@return: AUC
	@rtype: float
	"""
	np = len(target)
	n = len(ord_pred)
	area = 0.0
	
	for i in range(n):
		if ord_pred[i] in target:
			for j in range(i+1, n):
				if ord_pred[j] not in target:
					area += 1.0
					
	area /= float(np * (n - np))
	return area


# Calculates the average precision at k
def ap_k(ord_pred, target, k=100):
	if len(ord_pred) > k:
		ord_pred = ord_pred[:k]

	score = 0.0
	num_hits = 0.0

	for i,p in enumerate(ord_pred):
		if p in target:
			num_hits += 1.0
			score += num_hits / (i+1.0)

	if not target:
		return 1.0

	return score / min(len(target), k)


# Calculates the mean average precision at k
def map_k(ord_pred_all, target_all, k=10):
	return np.mean([ap_k(p, a, k) for p,a in zip(ord_pred_all, target_all)])


# Calculates the normalized discounted cumulative gain at k
def ndcg_k(ord_pred, target, k=100):
	k = min(k, len(ord_pred))
	idcg = idcg_k(k)
	dcg_k = sum([int(ord_pred[i] in target) / math.log(i+2, 2) for i in xrange(k)])
	return dcg_k / idcg


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
	res = sum([1.0/math.log(i+2, 2) for i in xrange(k)])
	if not res:
		return 1.0
	else:
		return res


def evaluate(rec, ts):
	tot = {"auc":0.0, "ndcg":0.0, "ap":0.0}
	
	for u in ts.users:
		pred = misc.sort(rec.get_scores(u), rec.data.get_items(u))
		t = ts.get_items(u)
		
		area = auc(pred, t)
		ndcg = ndcg_k(pred, t)
		ap = ap_k(pred, t)
		
		tot["auc"] += area
		tot["ndcg"] += ndcg
		tot["ap"] += ap
		
		#logging.info("%d = {auc:%s, ap:%s, ndcg:%s}" %(i, area, ap, ndcg))
		
	for metric in tot:
		tot[metric] /= float(len(ts.users))

	return tot
