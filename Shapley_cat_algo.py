'''
Cloud deployment of game theoretic categorical clustering using apache spark: An application to car recommendation
https://www.sciencedirect.com/science/article/pii/S2666827021000505#sec3
https://github.com/srimantacse/ShapleyCategorical/tree/main/src
'''
import math
import random
import numpy as np
from numpy import array
import time

def encode_features(X, enc_map=None):
    """Converts categorical values in each column of X to integers in the range
    [0, n_unique_values_in_column - 1], if X is not already of integer type.

    If mapping is not provided, it is calculated based on the values in X.

    Unknown values during prediction get a value of -1. np.NaNs are ignored
    during encoding, and get treated as unknowns during prediction.
    """
    if enc_map is None:
        fit = True
        # We will calculate enc_map, so initialize the list of column mappings.
        enc_map = []
    else:
        fit = False

    Xenc = np.zeros(X.shape).astype('int')
    for ii in range(X.shape[1]):
        if fit:
            col_enc = {val: jj for jj, val in enumerate(np.unique(X[:, ii]))
                       if not (isinstance(val, float) and np.isnan(val))}
            enc_map.append(col_enc)
        # Unknown categories (including np.NaNs) all get a value of -1.
        Xenc[:, ii] = np.array([enc_map[ii].get(x, -1) for x in X[:, ii]])

    return Xenc, enc_map

def pandas_to_numpy(x):
    return x.values if 'pandas' in str(x.__class__) else x

def list_max(l):
	if len(np.array(l).shape) == 2:
		#print("list_max: 2D List...")
		max_idx_y, max_val = list_max(l[0])
		max_idx_x = 0
		itr = 0
		for row in l:
			idx, val = list_max(row)
			if max_val < val:
				max_val = val
				max_idx_x = itr
				max_idx_y = idx
				
			itr = itr + 1
			
		return ([max_idx_x, max_idx_y], max_val)
	else:
		max_idx = np.argmax(l)
		max_val = l[max_idx]
		return (max_idx, max_val)

def point_distance(Z, X):
    """
    Calculates disssimilarity between any 2 records supplied by using Simple Matching Dissimilarity Measure algorithm.
    :param Z: Record 1
    :param X: Record 2
    :return: Dissimilarity between Z and X
    """
    m = len(Z)

    dissimlarity = 0

    for j in range(m):
        if Z[j] != X[j]:
            dissimlarity += 1

    return dissimlarity

def compute_num_dist(data1,data2):
	n1,d = np.shape(data1)
	n2,d = np.shape(data2)

	dist = np.zeros([n1,n2])
	for i in np.arange(n1):
		for j in np.arange(n2):
			dist[i][j] = np.sqrt(np.sum((data1[i] - data2[j])**2))

	return dist

def compute_catdist(data1,data2):
	if np.array(data1).ndim < 2:
		return point_distance(data1,data2)
		
	n1,d = np.shape(data1)
	n2,d = np.shape(data2)

	dist = np.zeros([n1,n2])
	for i in np.arange(n1):
		for j in np.arange(n2):
			dist[i][j] = point_distance(data1[i], data2[j])

	return dist

def compute_dist(data1,data2, isItCat=False):
	if isItCat:
		return compute_catdist(data1,data2)
	else:
		#cdist(center, data)
		return compute_num_dist(data1,data2)

def normDist(dist, dMax):
	return dist/(dMax + 1)
	
def inverseNormDist(dist, dMax):
	return 1 - normDist(dist, dMax)

def f(dist, dMax):
	return 1 -(dist/(dMax + 1))
	
def shapley(data, P=5):
	rowCount   = data.shape[0]
	dMax       = 0.0

	#print("\n>>================= Shapley ================================")
	#print("Info:: Shapley running with P: " + str(P) + " Data Shape: " + str(data.shape)) 
	shapley  = [0] * rowCount

	random.seed(time.time())
	randomDataIndex = random.sample(range(0, rowCount), P)

	# Find dMax, distance
	#print("Info:: Distance Calculation Started...")

	selectedData = [data[x] for x in randomDataIndex]
	distance     = compute_dist(data, selectedData, True)
	dMax         = np.amax(distance)
	#print("Info:: Calculatd Distance Shape: " + str(array(distance).shape))
	#print("Info:: Calculatd Distance Max: "   + "{:0,.2f}".format(dMax))	

	# Find Shapley
	#print("Info:: Shapley Calculation Started...")
	for i in range(rowCount):
		result = 0.0
		for j in range(len(randomDataIndex)):
			result = result + inverseNormDist(distance[i][j], dMax)
			
		shapley[i] = (result / 2.0)		
		
	#print("Info:: Shapley Calculation Done.")
	#print("-------------------------------------------------------------")

	return list(shapley), dMax

def shapley_cat(x, delta=.9, P=5):
	data, _ = encode_features(pandas_to_numpy(x))
	rowCount   = data.shape[0]
	dMax       = 0.0
	
	#print("\n=================== ShapleyCat ===================================")
	#print("Info:: ShapleyCat running with P: " + str(P) + " Data Shape: " + str(data.shape) + "  Delta: " + str(delta)) 
	_shapley, dMax  = shapley(data, P)
	
	# Find Cluster
	itr        = 0
	validIndex = [i for i in range(rowCount)]
	
	K          = []
	KIndex     = []
	Fi         = array(_shapley)
	Q          = data.tolist()
	
	#print("Info:: Clustering Started...")
	while True:
		itr = itr + 1
		
		x = array(validIndex)
		validList = x[x != -1]
		
		#print("--Iteration:\t" + str(itr) + "\tValidIndex No: " + str(len(validList)))
		
		if len(validList) == 0:
			break
		
		t, v = list_max(Fi)
		xt   = Q[t]
		
		# K = K U {xt}
		K.append(xt)
		KIndex.append(t)
		
		##print ("t: " + str(t)+ " K: " + str(K)+"\n")		
		for i in range(len(Q)):
			if Fi[i] == 0:
				continue
			xi = Q[i]
			dist = compute_dist(array(xi).reshape(1, -1), array(xt).reshape(1, -1), True)
			calc_dist = sum(f(dist, dMax))
			##print(calc_dist)
			if calc_dist >= delta:
				validIndex[i] = -1
				Q[i]          = []
				Fi[i]         = 0		
	
	#print("Info:: ShapleyCat Cluster Length:\t" + str(len(list(K))))
	#print("=================== ShapleyCat ===================================\n")
	return data, list(K), list(KIndex)