# -*- coding: utf-8 -*-
'''
A method for k-means-like clustering of categorical data
https://link.springer.com/article/10.1007/s12652-019-01445-5
This file is used to run the Modified 2 and Modified 3 by switching the parameter `use_global_attr_count`.
If `use_global_attr_count = 1`, it runs Modified 2.
Otherwise, it runs Modified 3.
'''

from collections import defaultdict
import numpy as np
import math

def vector_matching_dissim(centroid, a, global_attr_freq, w, beta):
    '''Get distance between a centroid and a'''
    distance = 0.
    for ic, curc in enumerate(centroid):
        d_ic = 0.
        keys = curc.keys()
        for key in keys:
            d_ic += (curc[key] * attr_dissim(key, a[ic], ic, global_attr_freq))

        d_ic *= math.pow(w[ic], beta)
        distance += d_ic
    return distance


def vectors_matching_dissim(vectors, a, global_attr_freq, w, beta):
    '''Get nearest vector in vectors to a'''
    min = np.Inf
    min_clust = -1
    for clust in range(len(vectors)):
        distance = vector_matching_dissim(vectors[clust], a, global_attr_freq, w, beta)
        if distance < min:
            min = distance
            min_clust = clust
    return min_clust, min


def attr_dissim(x, y, iattr, global_attr_freq):
    if (global_attr_freq[iattr][x] == 1.0) and (global_attr_freq[iattr][y] == 1.0):
        return 0
    if x == y:
        numerator = 2 * math.log(global_attr_freq[iattr][x])
    else:
        numerator = 2 * math.log((global_attr_freq[iattr][x] + global_attr_freq[iattr][y]))
    denominator = math.log(global_attr_freq[iattr][x]) + math.log(global_attr_freq[iattr][y])
    return 1 - numerator / denominator


def move_point_between_clusters(point, ipoint, to_clust, from_clust,
    cl_attr_freq, membership):
    '''Move point between clusters, categorical attributes'''
    membership[to_clust, ipoint] = 1
    membership[from_clust, ipoint] = 0
    # Update frequencies of attributes in clusters
    for iattr, curattr in enumerate(point):
        cl_attr_freq[to_clust][iattr][curattr] += 1
        cl_attr_freq[from_clust][iattr][curattr] -= 1
    return cl_attr_freq, membership


def labels_cost(X, centroids, global_attr_freq, w, beta):
    '''
    Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-modes algorithm.
    '''

    npoints = X.shape[0]
    labels = np.empty(npoints, dtype = 'int64')
    for ipoint, curpoint in enumerate(X):
        clust, diss = vectors_matching_dissim(centroids, curpoint, global_attr_freq, w, beta)
        labels[ipoint] = clust

    return labels


def cal_lambda(cl_attr_freq, clust_members):
    '''Re-calculate optimal bandwitch for each cluster'''
    if clust_members <= 1:
        return 0.
    numerator = 0.
    denominator = 0.
    for iattr, curattr in enumerate(cl_attr_freq):
        n_ = 0.
        d_ = 0.
        keys = curattr.keys()
        for key in keys:
            n_ += math.pow(1.0 * curattr[key] / clust_members, 2)
            d_ += math.pow(1.0 * curattr[key] / clust_members, 2)
        numerator += (1 - n_)
        denominator += (d_ - 1.0 / (len(keys)))
    return (1.0 * numerator) / ((clust_members - 1) * denominator)


def kmeans_like_iter(X, centroids, cl_attr_freq, membership, global_attr_freq, lbd, use_global_attr_count, w, beta):
    '''Single iteration of k-representative clustering algorithm'''
    moves = 0
    for ipoint, curpoint in enumerate(X):
        clust, distance = vectors_matching_dissim(centroids, curpoint, global_attr_freq, w, beta)
        if membership[clust, ipoint]:
            continue

        moves += 1
        old_clust = np.argwhere(membership[:, ipoint])[0][0]

        cl_attr_freq, membership = move_point_between_clusters(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membership)

        if sum(membership[old_clust, :]) == 0:
            from_clust = membership.sum(axis = 1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindex = np.random.choice(choices)

            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, old_clust, from_clust, cl_attr_freq, membership)

        for curc in (clust, old_clust):
            lbd[curc] = cal_lambda(cl_attr_freq[curc], sum(membership[curc, :]))

        for iattr in range(len(curpoint)):
            for curc in (clust, old_clust):
                cluster_members = sum(membership[curc, :])
                if use_global_attr_count:
                    centroids[curc][iattr] = cal_centroid_value(lbd[curc], cl_attr_freq[curc][iattr], cluster_members, len(global_attr_freq[iattr]))
                else:
                    attr_count = len(cl_attr_freq[curc][iattr].keys())
                    centroids[curc][iattr] = cal_centroid_value(lbd[curc], cl_attr_freq[curc][iattr], cluster_members, attr_count)

    return centroids, moves, lbd


def cal_global_attr_freq(X, npoints, nattrs):
    global_attr_freq = [defaultdict(float) for _ in range(nattrs)]

    for ipoint, curpoint in enumerate(X):
        for iattr, curattr in enumerate(curpoint):
            global_attr_freq[iattr][curattr] += 1.
    for iattr in range(nattrs):
        for key in global_attr_freq[iattr].keys():
            global_attr_freq[iattr][key] /= npoints

    return global_attr_freq


def cal_centroid_value(lbd, cl_attr_freq_attr, cluster_members, attr_count):
    '''Calculate centroid value at iattr'''
    assert cluster_members >= 1, "Cluster has no member, why?"

    keys = cl_attr_freq_attr.keys()
    vjd = defaultdict(float)
    for odl in keys:
        vjd[odl] = lbd / attr_count + (1 - lbd) * (1.0 * cl_attr_freq_attr[odl] / cluster_members)
    return vjd


def update_weights(X, centroids, membership, global_attr_freq, w, beta):
    nattrs = X.shape[1]
    D = [0. for _ in range(nattrs)]

    for iclust, curc in enumerate(centroids):
        for ipoint, curpoint in enumerate(X):
            for iattr in range(nattrs):
                for key in curc[iattr].keys():
                    if membership[iclust][ipoint]:
                        D[iattr] += membership[iclust][ipoint] * attr_dissim(curpoint[iattr], key, iattr, global_attr_freq)

    Dt = []
    for iattr in range(nattrs):
        if D[iattr] != 0.:
            Dt.append(D[iattr])

    for iattr in range(nattrs):
        if D[iattr] == 0.:
            w[iattr] = 0.
        else:
            denominator = 0.
            for t in range(len(Dt)):
                denominator += math.pow(D[iattr] / Dt[t], 1.0 / (beta - 1))
            w[iattr] = 1.0 / denominator
    return w


def kmeans_like(X, n_clusters, init, n_init, verbose, use_global_attr_count, beta):
    '''K-means-like algorithm'''

    X = np.asanyarray(X)
    npoints, nattrs = X.shape
    assert n_clusters < npoints, "More clusters than data points?"

    for init_no in range(n_init):
        #__INIT__
        if verbose:
            print("Clustering using GLOBAL attr count" if use_global_attr_count else "Clustering using LOCAL attr count")

        cl_attr_freq = [ [defaultdict(int) for _ in range(nattrs)] for _ in range(n_clusters)]
        membership = np.zeros((n_clusters, npoints), dtype = int)

        if init == 'Huang':
            initial_centroids_idx = np.random.choice(range(npoints), n_clusters, replace=False)
            centroids = [ [defaultdict(float) for _ in range(nattrs)] for _ in range(n_clusters)]
            for iclust in range(n_clusters):
                cur_centroid = X[initial_centroids_idx[iclust]]
                for iattr in range(nattrs):
                    centroids[iclust][iattr][cur_centroid[iattr]] = 1.0
        else:
            initial_centroids_idx = np.random.choice(range(npoints), n_clusters, replace=False)
            centroids = [ [defaultdict(float) for _ in range(nattrs)] for _ in range(n_clusters)]
            for iclust in range(n_clusters):
                for iattr in range(nattrs):
                    centroids[iclust][iattr] = cal_centroid_value(1., cl_attr_freq[iclust][iattr], npoints, 1)

        for ipoint, curpoint in enumerate(X):
            clust = ipoint % n_clusters
            membership[clust, ipoint] = 1
            for iattr, curattr in enumerate(curpoint):
                cl_attr_freq[clust][iattr][curattr] += 1

        global_attr_freq = cal_global_attr_freq(X, npoints, nattrs)
        lbd = [0. for _ in range(n_clusters)]
        w = np.ones(nattrs)

        for ii in range(100):
            centroids, moves, lbd = kmeans_like_iter(X, centroids, cl_attr_freq, membership, global_attr_freq, lbd, use_global_attr_count, w, beta)
            w = update_weights(X, centroids, membership, global_attr_freq, w, beta)

            if moves == 0:
                break

        labels = labels_cost(X, centroids, global_attr_freq, w, beta)
    return labels

if __name__ == "__main__":
    dataset= "soybean.csv"
    x = np.genfromtxt(dataset, dtype=int, delimiter=',')[:, :-1]
    k=4
    labels = kmeans_like(x, n_clusters=k, init='random', n_init=10, verbose=0, use_global_attr_count=0, beta=8)
    print(labels)