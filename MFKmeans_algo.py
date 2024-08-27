# -*- coding: utf-8 -*-
"""
Manhattan Frequency k-Means (MFk-Means) implementation with data preprocessing
https://www.sciencedirect.com/science/article/pii/S0045790617327131
https://github.com/vatsarishabh22/MFk-M-Clustering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ================================================================
# Data Preprocessing: Transform the CSV data based on frequencies
# ================================================================

def preprocess_data_to_array(x):
    rows, columns = x.shape
    field_to_value_to_count = [{} for _ in range(columns)]
    
    # First pass: Calculate frequencies of each value in each column
    for row in x:
        for i in range(columns):
            field_value = row[i]
            if field_value not in field_to_value_to_count[i]:
                field_to_value_to_count[i][field_value] = 0
            field_to_value_to_count[i][field_value] += 1

    # print("Total records: " + str(rows))

    # Second pass: Transform the data based on relative frequencies and store in a list
    transformed_data = []
    for row in x:
        transformed_values = []
        for i in range(columns):
            field_value = row[i]
            freq = field_to_value_to_count[i][field_value]
            relative_freq = float(freq) / 10000
            transformed_values.append(relative_freq)
        transformed_data.append(transformed_values)

    # Convert the list of transformed data to a NumPy array
    transformed_array = np.array(transformed_data, dtype=float)
    
    return transformed_array

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        for i in range(self.max_iter):
            cluster_group, wcss = self.assign_clusters(X)
            old_centroids = self.centroids
            self.centroids = self.move_centroids(X, cluster_group)

            if (np.array(old_centroids == self.centroids)).all():
                break

        return cluster_group, wcss

    def assign_clusters(self, X):
        cluster_group = []
        wcss = 0

        for row in X:
            distances = []
            for centroid in self.centroids:
                distances.append(np.dot(abs(row - centroid), 1).sum())
            min_distance = min(distances)
            wcss += (min_distance) ** 2
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)

        return np.array(cluster_group), wcss

    def move_centroids(self, X, cluster_group):
        new_centroids = []
        cluster_type = np.unique(cluster_group)

        for type in range(self.n_clusters):
            if type in cluster_type:
                new_centroids.append(X[cluster_group == type].mean(axis=0))
            else:
                # If a cluster is empty, reinitialize its centroid randomly
                new_centroids.append(X[random.randint(0, X.shape[0] - 1)])
        return np.array(new_centroids)

    def _inertia(self, X, n_clusters=2):
        km = KMeans(self.n_clusters, self.max_iter)
        _, dist_sqr = km.fit_predict(X)
        return dist_sqr

def mfk(x,k):
    transformed_data_path = preprocess_data_to_array(x)
    model = KMeans(n_clusters=k)
    labels, _ = model.fit_predict(transformed_data_path)
    return labels