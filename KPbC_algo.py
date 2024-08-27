#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
k-PbC: an improved cluster center initialization for categorical data clustering
https://link.springer.com/article/10.1007/s10489-020-01677-5
https://github.com/ClarkDinh/k-PbC
'''

from collections import defaultdict
import evaluation
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class FPNode:
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        if parent is not None:
            parent.children[item] = self

    def itempath_from_root(self):
        path = []
        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent
        path.reverse()
        return path

class FPTree:
    def __init__(self, rank=None):
        self.root = FPNode(None)
        self.nodes = defaultdict(list)
        self.cond_items = []
        self.rank = rank

    def conditional_tree(self, cond_item, minsup):
        branches = []
        count = defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count

        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}

        cond_tree = FPTree(rank)
        for idx, branch in enumerate(branches):
            branch = sorted([i for i in branch if i in rank], key=rank.get, reverse=True)
            cond_tree.insert_itemset(branch, self.nodes[cond_item][idx].count)
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

    def insert_itemset(self, itemset, count=1):
        node = self.root
        node.count += count

        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
            else:
                child_node = FPNode(item, count, node)
                self.nodes[item].append(child_node)
                node = child_node

    def is_path(self):
        return len(self.root.children) <= 1 and all(
            len(self.nodes[i]) <= 1 and not self.nodes[i][0].children for i in self.nodes
        )

    def print_status(self, count, colnames):
        cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ", ".join(cond_items)
        # print(f"\r{count} itemset(s) from tree conditioned on items ({cond_items})", end="\n")

class MFITree:
    class Node:
        def __init__(self, item, count=1, parent=None):
            self.item = item
            self.parent = parent
            self.children = {}
            if parent is not None:
                parent.children[item] = self

    def __init__(self, rank):
        self.root = self.Node(None)
        self.nodes = defaultdict(list)
        self.cache = []
        self.rank = rank

    def insert_itemset(self, itemset, count=1):
        node = self.root
        for item in itemset:
            if item in node.children:
                node = node.children[item]
            else:
                child_node = self.Node(item, count, node)
                self.nodes[item].append(child_node)
                node = child_node

    def contains(self, itemset):
        i = 0
        for item in reversed(self.cache):
            if self.rank[itemset[i]] < self.rank[item]:
                break
            if itemset[i] == item:
                i += 1
            if i == len(itemset):
                return True

        for basenode in self.nodes[itemset[0]]:
            i = 0
            node = basenode
            while node.item is not None:
                if self.rank[itemset[i]] < self.rank[node.item]:
                    break
                if itemset[i] == node.item:
                    i += 1
                if i == len(itemset):
                    return True
                node = node.parent

        return False

def fpmax_step(tree, minsup, mfit, colnames, max_len, verbose):
    count = 0
    items = list(tree.nodes.keys())
    largest_set = sorted(tree.cond_items + items, key=mfit.rank.get)
    if not largest_set:
        return

    if tree.is_path() and not mfit.contains(largest_set):
        count += 1
        largest_set.reverse()
        mfit.cache = largest_set
        mfit.insert_itemset(largest_set)
        if max_len is None or len(largest_set) <= max_len:
            support = (
                min([tree.nodes[i][0].count for i in items])
                if items
                else tree.root.count
            )
            yield support, largest_set

    if verbose:
        print_status(count, colnames)

    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        items.sort(key=tree.rank.get)
        for item in items:
            if mfit.contains(largest_set):
                return
            largest_set.remove(item)
            cond_tree = tree.conditional_tree(item, minsup)
            for support, mfi in fpmax_step(
                cond_tree, minsup, mfit, colnames, max_len, verbose
            ):
                yield support, mfi

def fpmax(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0):
    if min_support <= 0.0 or min_support > 1.0:
        raise ValueError("Invalid value for `min_support`. It must be in (0, 1].")

    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    tree, rank = setup_fptree(df, min_support)

    minsup = math.ceil(min_support * len(df))
    generator = fpmax_step(tree, minsup, MFITree(rank), colname_map, max_len, verbose)

    return generate_itemsets(generator, len(df), colname_map)

def setup_fptree(df, min_support):
    num_itemsets = len(df)
    is_sparse = hasattr(df, "sparse")
    if is_sparse:
        itemsets = df.sparse.to_coo().tocsr()
    else:
        itemsets = df.values

    item_support = np.array(np.sum(itemsets, axis=0) / num_itemsets).reshape(-1)
    items = np.nonzero(item_support >= min_support)[0]
    indices = item_support[items].argsort()
    rank = {item: i for i, item in enumerate(items[indices])}

    if is_sparse:
        itemsets.eliminate_zeros()

    tree = FPTree(rank)
    for i in range(num_itemsets):
        nonnull = (
            itemsets.indices[itemsets.indptr[i] : itemsets.indptr[i + 1]]
            if is_sparse
            else np.where(itemsets[i, :])[0]
        )
        itemset = [item for item in nonnull if item

 in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_itemset(itemset)

    return tree, rank

def generate_itemsets(generator, num_itemsets, colname_map):
    itemsets = [(sup / num_itemsets, frozenset(iset)) for sup, iset in generator]
    res_df = pd.DataFrame(itemsets, columns=["support", "itemsets"])

    if colname_map is not None:
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([colname_map[i] for i in x])
        )

    return res_df

def transform_data(X, sparse=False):
    unique_items = sorted(set(item for transaction in X for item in transaction))
    columns_mapping = {item: idx for idx, item in enumerate(unique_items)}

    if sparse:
        indptr = [0]
        indices = []
        for transaction in X:
            for item in set(transaction):
                col_idx = columns_mapping[item]
                indices.append(col_idx)
            indptr.append(len(indices))
        data = [True] * len(indices)
        array = csr_matrix((data, indices, indptr), dtype=bool)
    else:
        array = np.zeros((len(X), len(columns_mapping)), dtype=bool)
        for row_idx, transaction in enumerate(X):
            for item in transaction:
                col_idx = columns_mapping[item]
                array[row_idx, col_idx] = True

    columns = sorted(columns_mapping, key=columns_mapping.get)
    return array, columns

def Initiate_Centers_with_Transactions(result_df, k, dataset):
    map_Items_TIDs_df = preprocess_dataset_df(dataset)
    
    top_k_itemsets = result_df.sort_values(by=['itemsets','support'], ascending=[False, True]).head(k)
    #print("\n Top k maximal frequent itemsets:")
    #print(top_k_itemsets)

    initial_groups = []
    for itemset in top_k_itemsets['itemsets']:
        mfi = set(itemset)
        transaction_ids_for_mfi = find_transaction_ids_for_mfi_df(mfi, map_Items_TIDs_df)
        initial_groups.append({'pattern': mfi, 'transaction_ids': transaction_ids_for_mfi})

    return initial_groups

def preprocess_dataset_df(dataset):
    df = pd.DataFrame({'transaction_id': range(len(dataset)), 'items': dataset})
    df_exploded = df.explode('items')
    df_exploded['items'] = df_exploded['items'].astype(str)
    return df_exploded.groupby('items')['transaction_id'].apply(set).to_dict()

def find_transaction_ids_for_mfi_df(mfi, map_Items_TIDs_df):
    transaction_ids = None
    for item in mfi:
        item_transaction_ids = map_Items_TIDs_df.get(item)
        if item_transaction_ids is not None:
            if transaction_ids is None:
                transaction_ids = item_transaction_ids.copy()
            else:
                transaction_ids.intersection_update(item_transaction_ids)
    return transaction_ids

# def remove_overlapping_transactions(initial_groups):
#     common_transaction_ids = set.intersection(*[cluster['transaction_ids'] for cluster in initial_groups])

#     for cluster in initial_groups:
#         cluster['transaction_ids'].difference_update(common_transaction_ids)

#     transaction_counts = {i: len(cluster['transaction_ids']) for i, cluster in enumerate(initial_groups)}

#     for i, cluster in enumerate(initial_groups):
#         cluster_pattern = cluster['pattern']
#         cluster_transaction_ids = cluster['transaction_ids']
#         overlapping_transactions = set()

#         for j, other_cluster in enumerate(initial_groups):
#             if i != j:
#                 other_transaction_ids = other_cluster['transaction_ids']
#                 overlapping = cluster_transaction_ids.intersection(other_transaction_ids)
#                 overlapping_transactions.update(overlapping)

#         for j, other_cluster in enumerate(initial_groups):
#             if i != j and overlapping_transactions.intersection(other_cluster['transaction_ids']):
#                 other_transaction_ids = other_cluster['transaction_ids']
#                 other_count = transaction_counts[j]
#                 current_count = transaction_counts[i]

#                 if current_count < other_count or (current_count == other_count and i < j):
#                     other_transaction_ids.difference_update(overlapping_transactions)
#                 else:
#                     cluster_transaction_ids.difference_update(overlapping_transactions)

#     return initial_groups

def remove_overlapping_transactions(initial_groups):
    # Identify common transactions that appear in all clusters
    common_transaction_ids = set.intersection(*[cluster['transaction_ids'] for cluster in initial_groups])

    # Remove common transactions from each cluster
    for cluster in initial_groups:
        cluster['transaction_ids'].difference_update(common_transaction_ids)

    # Track the initial count of transactions in each cluster
    transaction_counts = {i: len(cluster['transaction_ids']) for i, cluster in enumerate(initial_groups)}

    # Keep a backup of the original transactions in each cluster
    backup_transaction_ids = {i: cluster['transaction_ids'].copy() for i, cluster in enumerate(initial_groups)}

    for i, cluster in enumerate(initial_groups):
        cluster_transaction_ids = cluster['transaction_ids']
        overlapping_transactions = set()

        # Find overlapping transactions with other clusters
        for j, other_cluster in enumerate(initial_groups):
            if i != j:
                other_transaction_ids = other_cluster['transaction_ids']
                overlapping = cluster_transaction_ids.intersection(other_transaction_ids)
                overlapping_transactions.update(overlapping)

        # Remove overlapping transactions based on cluster sizes
        for j, other_cluster in enumerate(initial_groups):
            if i != j and overlapping_transactions.intersection(other_cluster['transaction_ids']):
                other_transaction_ids = other_cluster['transaction_ids']
                other_count = transaction_counts[j]
                current_count = transaction_counts[i]

                if current_count < other_count or (current_count == other_count and i < j):
                    other_transaction_ids.difference_update(overlapping_transactions)
                else:
                    cluster_transaction_ids.difference_update(overlapping_transactions)

        # Check if the current cluster is empty after removal
        if not cluster_transaction_ids:
            # Restore some transactions from the backup to ensure the cluster is not empty
            cluster['transaction_ids'] = backup_transaction_ids[i]
            
            # Remove restored transactions from the largest other cluster
            largest_cluster_index = max(transaction_counts, key=transaction_counts.get)
            if largest_cluster_index != i:
                largest_cluster = initial_groups[largest_cluster_index]
                transactions_to_remove = backup_transaction_ids[i]
                largest_cluster['transaction_ids'].difference_update(transactions_to_remove)

    return initial_groups

def calculate_relative_frequency(cluster_transactions):
    attribute_counts = [{} for _ in range(len(cluster_transactions[0]))]

    for transaction in cluster_transactions:
        for i, attribute in enumerate(transaction):
            if attribute in attribute_counts[i]:
                attribute_counts[i][attribute] += 1
            else:
                attribute_counts[i][attribute] = 1

    relative_frequency = [{category: count / len(cluster_transactions) for category, count in attribute_count.items()} for attribute_count in attribute_counts]
    return relative_frequency

def form_cluster_representatives(cluster_transactions):
    relative_frequencies = calculate_relative_frequency(cluster_transactions)
    cluster_representatives = []

    for attribute_frequencies in relative_frequencies:
        cluster_representative = [(category, frequency) for category, frequency in attribute_frequencies.items()]
        cluster_representatives.append(cluster_representative)

    return cluster_representatives

def calculate_distance_to_representatives(data_instance, cluster_representatives):
    distances = []
    for representative in cluster_representatives:
        distance = 0.0
        for i, attribute in enumerate(data_instance):
            for category, frequency in representative[i]:
                if category == attribute:
                    distance += frequency
        distances.append(distance)
    return distances

def assign_to_nearest_cluster(data_instances, cluster_representatives):
    assigned_clusters = []
    for data_instance in data_instances:
        distances = calculate_distance_to_representatives(data_instance, cluster_representatives)
        nearest_cluster = distances.index(max(distances))
        assigned_clusters.append(nearest_cluster)
    return assigned_clusters

def initial_assignment(initial_groups, dataset, output_file):
    cluster_transactions = []

    transaction_ids_list = [group['transaction_ids'] for group in initial_groups]

    for transaction_ids in transaction_ids_list:
        cluster_transaction = []
        for transaction_id in transaction_ids:
            cluster_transaction.append(dataset[transaction_id])
        cluster_transactions.append(cluster_transaction)

    if not cluster_transactions:
        #print("Error: No cluster transactions found.")
        return

    cluster_representatives = [form_cluster_representatives(cluster_transactions) for cluster_transactions in cluster_transactions]

    assigned_clusters = assign_to_nearest_cluster(dataset, cluster_representatives)

    #print("\n Assigned Clusters:")
    # for i, cluster_id in enumerate(assigned_clusters, 1):
    #     print(f"Data instance {i}: cluster {cluster_id + 1}")
    
    # Extract clusters from assigned_clusters
    clusters = [[] for _ in range(len(initial_groups))]
    for i, cluster_id in enumerate(assigned_clusters):
        clusters[cluster_id].append(i)

    # Save clusters to a text file
    with open(output_file, 'w') as file:
        for cluster in clusters:
            file.write(' '.join(str(tid) for tid in cluster) + '\n')

# -------------------------Main steps of K-PbC--------------------------------------------#

def categorical_dissimilarity(x, y, attribute_index, global_attr_freq):
    '''Calculate dissimilarity between two categorical attributes'''
    if (global_attr_freq[attribute_index][x] == 1.0) and (global_attr_freq[attribute_index][y] == 1.0):
        return 0
    if x == y:
        numerator = 2 * np.log(global_attr_freq[attribute_index][x])
    else:
        numerator = 2 * np.log((global_attr_freq[attribute_index][x] + global_attr_freq[attribute_index][y]))
    denominator = np.log(global_attr_freq[attribute_index][x]) + np.log(global_attr_freq[attribute_index][y])
    return 1 - numerator / denominator

def vector_dissimilarity(centroid, vector, global_attr_freq):
    '''Calculate distance between a centroid and a vector'''
    distance = 0.
    for attribute_index, cur_centroid in enumerate(centroid):
        for key, value in cur_centroid.items():
            distance += value * categorical_dissimilarity(key, vector[attribute_index], attribute_index, global_attr_freq)
    return distance

def nearest_vector(centroids, vector, global_attr_freq):
    '''Get nearest vector in centroids to the given vector'''
    min_distance = np.Inf
    nearest_cluster = -1
    for cluster_index, centroid in enumerate(centroids):
        distance = vector_dissimilarity(centroid, vector, global_attr_freq)
        if distance < min_distance:
            min_distance = distance
            nearest_cluster = cluster_index
    return nearest_cluster, min_distance

def move_point_between_clusters(point, point_index, to_cluster, from_cluster,
    cluster_attr_freq, membership):
    '''Move point between clusters and update frequencies'''
    membership[to_cluster, point_index] = 1
    membership[from_cluster, point_index] = 0
    for attribute_index, attribute_value in enumerate(point):
        cluster_attr_freq[to_cluster][attribute_index][attribute_value] += 1
        cluster_attr_freq[from_cluster][attribute_index][attribute_value] -= 1
    return cluster_attr_freq, membership

def calculate_lambda(cluster_attr_freq, cluster_members):
    numerator = 0.
    denominator = 0.

    for attribute_freq in cluster_attr_freq:
        n_ = sum((1.0 * freq / cluster_members) ** 2 for freq in attribute_freq.values())
        d_ = sum((1.0 * freq / cluster_members) ** 2 for freq in attribute_freq.values())
        numerator += (1 - n_)
        denominator += (d_ - 1.0 / len(attribute_freq))

    # Handle division by zero or cluster_members == 1
    if denominator == 0 or cluster_members == 1:
        # Use a default value for lambda
        lambda_value = 0.2
    else:
        lambda_value = (1.0 * numerator) / ((cluster_members - 1) * denominator)

    return lambda_value


def calculate_centroid_value(lbd, cluster_attr_freq_attr, cluster_members, attribute_count):
    '''Calculate centroid value at attribute_index'''
    assert cluster_members >= 1, "Cluster has no member."

    centroid_value = defaultdict(float)
    for attribute_value, freq in cluster_attr_freq_attr.items():
        centroid_value[attribute_value] = lbd / attribute_count + (1 - lbd) * (1.0 * freq / cluster_members)
    return centroid_value

def k_PbC_iteration(X, centroids, cluster_attr_freq, membership, global_attr_freq, lbd):
    '''Perform one iteration of k-PbC algorithm'''
    for point_index, point in enumerate(X):
        cluster_index, distance = nearest_vector(centroids, point, global_attr_freq)
        if membership[cluster_index, point_index]:
            continue

        old_cluster_indices = np.argwhere(membership[:, point_index])
        if len(old_cluster_indices) == 0:  # No non-zero elements
            continue
        old_cluster = old_cluster_indices[0][0]

        # old_cluster = np.argwhere(membership[:, point_index])[0][0]
        cluster_attr_freq, membership = move_point_between_clusters(
            point, point_index, cluster_index, old_cluster, cluster_attr_freq, membership)
        if sum(membership[old_cluster, :]) == 0:
            from_cluster = np.argmax(membership.sum(axis=1))
            choices = [index for index, choice in enumerate(membership[from_cluster, :]) if choice]
            random_index = np.random.choice(choices)
            cluster_attr_freq, membership = move_point_between_clusters(
                X[random_index], random_index, old_cluster, from_cluster, cluster_attr_freq, membership)
        for current_cluster in (cluster_index, old_cluster):
            lbd[current_cluster] = calculate_lambda(cluster_attr_freq[current_cluster], sum(membership[current_cluster, :]))
        for attribute_index in range(len(point)):
            for current_cluster in (cluster_index, old_cluster):
                cluster_members = sum(membership[current_cluster, :])
                attribute_count = len(cluster_attr_freq[current_cluster][attribute_index].keys())
                centroids[current_cluster][attribute_index] = calculate_centroid_value(lbd[current_cluster], cluster_attr_freq[current_cluster][attribute_index], cluster_members, attribute_count)
    #print("Centroids:",centroids)
    #print("Lambda:",lbd)
    return centroids, lbd

def calculate_labels(X, centroids, global_attr_freq):
    '''Calculate labels given centroids'''
    npoints = len(X)
    nattrs = len(X[0])
    labels = np.empty(npoints, dtype='int64')
    for point_index, point in enumerate(X):
        cluster_index, dissimilarity = nearest_vector(centroids, point, global_attr_freq)
        assert cluster_index != -1, "No cluster found."
        labels[point_index] = cluster_index
    return labels

def read_cluster():
    '''Read cluster data from file'''
    ifile = "initial_clusters.txt"
    centroids = []
    with open(ifile) as fp:
        for index, line in enumerate(fp):
            if line == "\n":
                break
            points = line.split()
            centroids.append(points)
    return centroids

def calculate_global_attr_freq(X, npoints, nattrs):
    '''Calculate global attribute frequencies'''
    global_attr_freq = [defaultdict(float) for _ in range(nattrs)]
    for point_index, point in enumerate(X):
        for attribute_index, attribute_value in enumerate(point):
            global_attr_freq[attribute_index][attribute_value] += 1.
    for attribute_index in range(nattrs):
        for key in global_attr_freq[attribute_index].keys():
            global_attr_freq[attribute_index][key] /= npoints
    return global_attr_freq

def initialize_clusters(X, n_clusters, n_attrs, n_points):
    '''Initialize clusters'''
    cluster_attr_freq = [[defaultdict(int) for _ in range(n_attrs)]
                    for _ in range(n_clusters)]
    membership = np.zeros((n_clusters, n_points), dtype='int64')
    centroids = [[defaultdict(float) for _ in range(n_attrs)]
                 for _ in range(n_clusters)]
    initial_clusters = read_cluster()
    #print(initial_clusters)
    for cluster_index in range(n_clusters):
        for point_index in range(len(initial_clusters[cluster_index])):
            membership[cluster_index, int(initial_clusters[cluster_index][point_index])] = 1
            for attribute_index, attribute_value in enumerate(X[int(initial_clusters[cluster_index][point_index])]):
                cluster_attr_freq[cluster_index][attribute_index][attribute_value] += 1
    return membership, cluster_attr_freq, centroids

def k_PbC(X, k):
    '''Perform k-PbC algorithm'''
    npoints = len(X)
    nattrs = len(X[0])
    global_attr_freq = calculate_global_attr_freq(X, npoints, nattrs)
    membership, cluster_attr_freq, centroids = initialize_clusters(
        X, k, nattrs, npoints)

    lbd = [0.5 for _ in range(k)]
    centroids, lbd = k_PbC_iteration(X, centroids, cluster_attr_freq, membership,
                                          global_attr_freq, lbd)

    labels = calculate_labels(X, centroids, global_attr_freq)
    return labels

# Define a function to preprocess each column
def preprocess_column(column, col_index):
    categorical_values = set(column)
    value_mapping = {value: f"{value}_{col_index}" for value in categorical_values}
    return column.map(value_mapping)

def k_PbC_clustering(x_df,k,min_support):
    # Preprocess each column except the last one (assuming the last column is the target column)
    for col_index, column in enumerate(x_df.columns, start=1):
        x_df[column] = preprocess_column(x_df[column], col_index)
    
    X = x_df.values
    X = X.tolist()
    transformed_data, columns = transform_data(X, sparse=False)
    df = pd.DataFrame(transformed_data, columns=columns)
    mfi_df = fpmax(df, min_support, use_colnames=True, max_len=None, verbose=0)
    
    if mfi_df.empty or mfi_df.shape[0] < k:
        raise RuntimeError("Cannot find enough clusters at minsup={}. Please lower the minSup threshold!".format(min_support))

    #print("List of maximal frequent itemsets at minsup={}:".format(min_support))
    #print(mfi_df)

    initial_groups = Initiate_Centers_with_Transactions(mfi_df, k, X)
    initial_groups = remove_overlapping_transactions(initial_groups)

    output_file = "initial_clusters.txt"
    initial_assignment(initial_groups, X, output_file)

    labels = k_PbC(X, k)
    return labels