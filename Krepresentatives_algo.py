'''
An alternative extension of the k-means algorithm for clustering categorical data
@article{san2004alternative,
  title={An alternative extension of the k-means algorithm for clustering categorical data},
  author={San, Ohn Mar and Huynh, Van-Nam and Nakamori, Yoshiteru},
  journal={International journal of applied mathematics and computer science},
  volume={14},
  number={2},
  pages={241--247},
  year={2004},
  publisher={Uniwersytet Zielonog{\'o}rski. Oficyna Wydawnicza}
}
'''
from __future__ import division
from collections import defaultdict
import numpy as np

def get_max_value_key(dic):
    """Fast method to get key for maximum value in dict."""
    v = list(dic.values())
    k = list(dic.keys())
    return k[v.index(max(v))]

def attr_dissim(x, y):
    """Dissimilarity between 2 categorical attributes x and y."""
    return 0 if x == y else 1

def vector_matching_dissim(centroid, a):
    """Get distance between a centroid and a."""
    return sum(centroid[i][key] * attr_dissim(key, a[i]) for i in range(len(centroid)) for key in centroid[i])

def vectors_matching_dissim(vectors, a):
    """Get nearest vector in vectors to a."""
    distances = [vector_matching_dissim(vectors[clust], a) for clust in range(len(vectors))]
    min_clust = np.argmin(distances)
    return min_clust, distances[min_clust]

def move_point_between_clusters(point, ipoint, to_clust, from_clust,
    cl_attr_freq, membership):
    """Move point between clusters, categorical attributes."""
    membership[to_clust, ipoint] = 1
    membership[from_clust, ipoint] = 0
    for iattr, curattr in enumerate(point):
        cl_attr_freq[to_clust][iattr][curattr] += 1
        cl_attr_freq[from_clust][iattr][curattr] -= 1
    return cl_attr_freq, membership

def matching_dissim(a, b):
    """Simple matching dissimilarity function."""
    return np.sum(a != b, axis=1)

def _init_clusters(X, centroids, n_clusters, nattrs, npoints, verbose):
    """Initialize clusters."""
    membership = np.zeros((n_clusters, npoints), dtype='int64')
    cl_attr_freq = [[defaultdict(int) for _ in range(nattrs)]
                    for _ in range(n_clusters)]
    total_membership = membership.sum(axis=1)
    for ipoint, curpoint in enumerate(X):
        clust = np.argmin(matching_dissim(centroids, curpoint))
        membership[clust, ipoint] = 1
        for iattr, curattr in enumerate(curpoint):
            cl_attr_freq[clust][iattr][curattr] += 1
        total_membership[clust] += 1
    for ik, cluster_members in enumerate(total_membership):
        if cluster_members == 0:
            from_clust = total_membership.argmax()
            choices = np.where(membership[from_clust, :])[0]
            rindex = np.random.choice(choices)
            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, ik, from_clust, cl_attr_freq, membership)
    return cl_attr_freq, membership

def cal_centroid_value(cl_attr_freq_attr, cluster_members):
    """Calculate centroid value at iattr."""
    return defaultdict(float, {key: cl_attr_freq_attr[key] / cluster_members for key in cl_attr_freq_attr})

def _k_presentative_iter(X, centroids, cl_attr_freq, membership):
    """Single iteration of k-representative clustering algorithm."""
    moves = 0
    for ipoint, curpoint in enumerate(X):
        clust, distance = vectors_matching_dissim(centroids, curpoint)
        if membership[clust, ipoint]:
            continue
        moves += 1
        old_clust = np.argwhere(membership[:, ipoint])[0][0]
        cl_attr_freq, membership = move_point_between_clusters(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membership)
        if membership[old_clust].sum() == 0:
            from_clust = np.argmax(membership.sum(axis=1))
            choices = np.where(membership[from_clust, :])[0]
            rindex = np.random.choice(choices)
            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, old_clust, from_clust, cl_attr_freq, membership)
        for iattr in range(len(curpoint)):
            for curc in (clust, old_clust):
                cluster_members = membership[curc].sum()
                centroids[curc][iattr] = cal_centroid_value(cl_attr_freq[curc][iattr], cluster_members)
    return centroids, moves

def _labels_cost(X, centroids):
    """Calculate labels and cost function."""
    npoints, nattrs = X.shape
    cost = 0.
    labels = np.empty(npoints, dtype='int64')
    for ipoint, curpoint in enumerate(X):
        clust, diss = vectors_matching_dissim(centroids, curpoint)
        labels[ipoint] = clust
        cost += diss
    return labels, cost

def k_representatives(X, n_clusters, init, n_init, verbose):
    """k-representatives algorithm."""
    X = np.asanyarray(X)
    npoints, nattrs = X.shape
    assert n_clusters < npoints, "More clusters than data points?"
    all_centroids = []
    all_labels = []
    all_costs = []
    if init == 'random':
        seeds = np.random.choice(range(npoints), n_clusters)
        centroids = X[seeds]
    else:
        raise NotImplementedError
    cl_attr_freq, membership = _init_clusters(X, centroids, n_clusters, nattrs, npoints, verbose)
    centroids = [[defaultdict(float) for _ in range(nattrs)]
                 for _ in range(n_clusters)]
    total_membership = membership.sum(axis=1)
    for ik, cluster_members in enumerate(total_membership):
        for iattr in range(nattrs):
            centroids[ik][iattr] = cal_centroid_value(cl_attr_freq[ik][iattr], cluster_members)
    converged = False
    cost = np.Inf
    while not converged:
        centroids, moves = _k_presentative_iter(X, centroids, cl_attr_freq, membership)
        labels, ncost = _labels_cost(X, centroids)
        converged = (moves == 0)
        cost = ncost
    all_centroids.append(centroids)
    all_labels.append(labels)
    all_costs.append(cost)
    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}, min cost is {}" . format(best + 1, all_costs[best]))
    return all_centroids[best], all_labels[best], all_costs[best]

class KRepresentatives(object):
    """k-representative clustering algorithm for categorical data."""
    def __init__(self, n_clusters, n_init, init='random', max_iter=1, verbose=1):
        if verbose:
            print("Number of clusters: {}".format(n_clusters))
            print("Init type: {}".format(init))
            print("Local loop: {}".format(n_init))
            print("Max iterations: {}".format(max_iter))
        if hasattr(init, '__array__'):
            n_clusters = init.shape[0]
            init = np.asarray(init, dtype=np.float64)
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, **kwargs):
        """Compute k-representative clustering."""
        self.cluster_centroids_, self.labels_, self.cost_ = \
            k_representatives(X, self.n_clusters, self.init, self.n_init, self.verbose)
        return self

    def fit_predict(self, X, **kwargs):
        """Compute cluster centroids and predict cluster index for each sample."""
        return self.fit(X, **kwargs).labels_

    def predict(self, X, **kwargs):
        """Predict the closest cluster each sample in X belongs to."""
        return self.fit(X, **kwargs).labels_