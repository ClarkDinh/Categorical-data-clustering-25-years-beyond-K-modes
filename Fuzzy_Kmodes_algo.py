'''
A fuzzy k-modes algorithm for clustering categorical data https://ieeexplore.ieee.org/document/784206
This implementation is taken from https://github.com/srimantacse/ShapleyCategorical/tree/main/src
'''

import numpy as np
import pandas as pd
import operator
from collections import Counter
# from time import time
# import kmodes as km
import random

def pandas_to_numpy(x):
    return x.values if 'pandas' in str(x.__class__) else x

def initialize_centroids(X, n_clusters=4):
    """
    Performs selection of initial centroids (From kmodes as of now)
    :param X: The dataset of points to choose from
    :param n_clusters: number of initial points to choose and return
    :return: n_clusters initial points selected from X as per the algorithm used
    """

    # centroids, belongs_to = km.kmodes(X, n_clusters, debug=False)

    # return centroids
    return np.array(random.sample(list(X), n_clusters))

def calculate_dissimilarity(Z, X):
	"""
	Calculates disssimilarity between any 2 records supplied by using Simple Matching Dissimilarity Measure algorithm.
	:param Z: Record 1
	:param X: Record 2
	:return: Dissimilarity between Z and X
	"""
	m = len(list(Z))

	dissimlarity = 0

	for j in range(m):
		if Z[j] != X[j]:
			dissimlarity += 1

	return dissimlarity

def calculate_cost(W, Z, X, alpha):
    """
    Calculates the cost function of k-modes algorithm as per Huang '99 paper on fuzzy k-modes.
    :param W: Fuzzy partition matrix
    :param Z: Cluster centroids
    :param alpha: Weighing exponent
    :return: Cost of of the current setup using the formula from the paper
    """
    k = W.shape[0]
    n = W.shape[1]

    cost = 0

    for l in range(k):
        for i in range(n):
            cost += pow(W[l][i], alpha) * calculate_dissimilarity(Z[l], X[i])

    return cost

def calculate_partition_matrix(Z, X, alpha):
    """
    Calculates the dissimilarity matrix W for a fixed Z as per Theorem 1 in Huang '99 paper on fuzzy kmodes.
    :param Z: Fixed centroids
    :param X: Dataset points
    :param alpha: Weighing exponent
    :return: Dissimilarity matrix of type Numpy array of dimension k x n.
    """
    k = len(list(Z))

    n = X.shape[0]

    exponent = 1 / (alpha - 1)

    W = np.zeros((k, n), dtype='float')

    for l in range(k):
        for i in range(n):
            dli = calculate_dissimilarity(Z[l], X[i])
            if dli == 0:
                W[l][i] = 1
            else:
                flag = False
                sum = 0
                for h in range(k):
                    dhi = calculate_dissimilarity(Z[h], X[i])

                    if h != l and dhi == 0:
                        W[l][i] = 0
                        flag = True
                        break

                    sum += pow(dli / dhi, exponent)

                if not flag:
                    W[l][i] = 1 / sum

    return W

def calculate_centroids(W, X, alpha):
    """
    Calculates the updated value of Z as per Theorem 4 of paper by Huang '99 on fuzzy kmodes.
    :param W: Partition matrix
    :param X: Dataset
    :param alpha: Weighing exponent
    :return: Updated centroid Numpy matrix of dimension k x n.
    """
    k = W.shape[0]
    m = X.shape[1]

    Z = [[None] * m for i in range(k)]

    for l in range(k):
        for j in range(m):
            weights = []
            x_j = X[:, j]
            dom_aj = np.unique(x_j)
            for key in dom_aj.__iter__():
                indexes = [i for i in range(len(x_j)) if x_j[i] == key]
                sum = 0
                for index in indexes:
                    sum += pow(W[l][index], alpha)
                weights.append((key, sum))
            Z[l][j] = max(weights, key=operator.itemgetter(1))[0]

    return Z

def calculate_accuracy(labels, prediction):
    labels_values = np.unique(labels)
    count = 0.0

    for key in labels_values.__iter__():
        indices = [prediction[i] for i in range(len(prediction)) if labels[i] == key]
        count += max(Counter(indices).items(), key=operator.itemgetter(1))[1]

    return round(count / len(prediction), 4) * 100

def calculate_cluster_allotment(W):
    """
    Calculates the membership of each point to various clusters.
    :param W: Partition matrix
    :return: allotment array of dimension 1xn
    """
    n = W.shape[1]

    allotment = np.zeros(n, dtype='int')

    for i in range(n):
        allotment[i] = np.argmax(W[:, i]) + 1

    # for ii in range(7):
    #     if ii - 1 not in allotment:
    #         print W
    #         print allotment

    return allotment
	
def fuzzy_kmodes(X, initialCluster, n_clusters=4, alpha=1.1):
    """
    Calculates the optimal cost, cluster centers and fuzzy partition matrix for the given dataset.
    :param X: Dataset
    :param n_clusters: number of clusters to form
    :param alpha: Weighing exponent
    :return:
    """
    # t0 = time()

    if initialCluster == []:
        Z = initialize_centroids(X, n_clusters)
        # print('Info:: FKMode Cluster Seed: Random')
    else:
        Z = initialCluster
        #print('Info:: FKMode Cluster Seed: Provided')

    init = [Z]

    W = calculate_partition_matrix(Z, X, alpha)
    #f_old = calculate_cost(W, Z, X, alpha)

    f_new = 0

    while True:
        Z = calculate_centroids(W, X, alpha)
        init.append(Z)
        #f_new = calculate_cost(W, Z, X, alpha)

        #if f_new == f_old:
            #break

        f_old = f_new
        W = calculate_partition_matrix(Z, X, alpha)
        f_new = calculate_cost(W, Z, X, alpha)

        if f_new == f_old:
            break

    assigned_clusters = calculate_cluster_allotment(W)

    # t1 = round(time() - t0, 3)

    return f_new, Z, assigned_clusters, W

class FKModes():
    def __init__(self, init_cluster= [], n_clusters = 8, max_iter = 100, verbose = 0):

        self.n_clusters     = n_clusters
        self.init_clusters  = init_cluster
        self.max_iter       = max_iter
        self.verbose        = verbose
        self.cost           = 0.0
        self.n_iter_        = 0
        self.cluster_centroids = None
        self.labels            = None
        self.Um                = None

    def fit(self, X):
        X = pandas_to_numpy(X)
        alpha = 1.1
        self.cost, self.cluster_centroids, self.labels, self.Um = fuzzy_kmodes(X, self.init_clusters, self.n_clusters, alpha)
        
        return self.cost, self.cluster_centroids, self.labels, self.Um