
''''
Mutual Information Scoring: Increasing Interpretability in High-stakes Categorical Clustering Tasks 
https://link.springer.com/chapter/10.1007/978-3-031-17114-7_16
https://github.com/c4sgub/MIS_Categorical_Clustering
'''
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score, mutual_info_score
from collections import defaultdict

def attr_cond_prob(attr_col, class_col):
    contingency_table = pd.crosstab(attr_col, class_col)
    N_attr, N_classes = contingency_table.shape

    categories = list(contingency_table.index)
    classes = list(contingency_table.columns)

    n_cat = contingency_table.sum(axis=1)

    prob_tab = defaultdict(dict)
    for cat in categories:
        for cl in classes:
            n_xy = contingency_table.loc[cat, cl]
            prob_tab[cat][cl] = n_xy/n_cat.loc[cat]
    return prob_tab


def prob_list(data, class_col):
    '''list of conditional prob. len(attributes)'''
    prob_map = []
    for _, attr in data.iteritems():
        prob_map.append(attr_cond_prob(attr, class_col))
    return prob_map


def obj_score(row, cls, prob_map):
    N = row.shape[0]
    score = 0
    for i, at in enumerate(row):
        score += prob_map[i][at][cls]
    return score/N


def get_cluster_idx(data, decision_attr, category, vector=0, thresh=0):
    '''
    Args:
        data: DataFrame of a cluster
        decision_attr: Pandas Series of the splitting attribute

    Returns:
        idx: list of object indices belonging to the cluster
    '''
    if vector:
        comma_sep_series = data.apply(lambda x: ','.join(x.astype(str)), axis=1)
        object_vectors = pd.DataFrame({'comma_sep': comma_sep_series})
    else:
        object_vectors = data

    cat_vector = object_vectors.loc[decision_attr == category]
    indices = cat_vector.index
    
    if thresh == 0:
        return list(indices)
    
    p_map = prob_list(object_vectors, decision_attr)
    idx = []
    i = 0
    categories = decision_attr.unique()
    for _, rws in cat_vector.iterrows():
        temp = []
        temp_cls = obj_score(rws, category, p_map)
        for cls in categories:
            if cls != category:
                temp.append(obj_score(rws, cls, p_map))
        if len(temp) == 0 or temp_cls >= max(temp):
            idx.append(indices[i])
        i+= 1
    return idx

def entropy(p):
    '''
    Compute entropy of a discrete random variable

    Args: 
        p: probability distribution of the random variable

    Returns:
        entropy
    '''
    return sum(-1 * p[i] * math.log2(p[i]) for i in range(len(p)))

def information_measure(X, Y):
    '''
    Mutual Information between X and Y 

    Args:
        X, Y: Dataframe Series 

    Returns:
        mutual information
    '''
    # generate contigency table (X intersection Y)
    contingency_table = pd.crosstab(X, Y)
    N_x, N_y = contingency_table.shape
    N = X.shape[0]

    p_x = contingency_table.sum(axis=1)/N
    p_y = contingency_table.sum(axis=0)/N

    info_measure = 0
    h_x = 0
    h_y = 0
    for i in range(N_x):
        h_x += -1 * p_x.iloc[i] * math.log2(p_x.iloc[i])
        for j in range(N_y):
            h_y += -1 * p_y.iloc[j] * math.log2(p_y.iloc[j])
            p_xy = contingency_table.iloc[i, j]/N
            # if x and y belonging to random variable X and Y are disjoint
            if p_xy:
                info_measure += p_xy * \
                    math.log2(p_xy/(p_x.iloc[i]*p_y.iloc[j]))
                
    NMI = 2 * info_measure/(h_x + h_y)        
    return NMI

def NMI(X, Y):
    return mutual_info_score(X, Y)

def dataset_information_measure(dataset):
    '''
    Mutual Information for the dataset

    Args:
        dataset: Dataframe 

    Returns: 
        MI matrix
    '''
#     ncat = [np.unique(dataset[attr]).shape[0] for attr in dataset.columns]
    info_mat = np.empty(shape=(dataset.shape[1], dataset.shape[1]))
    for i, c1 in enumerate(dataset.columns):
        for j, c2 in enumerate(dataset.columns):
            inf_mea = NMI(dataset[c1], dataset[c2])
            info_mat[i, j] = inf_mea 
    return info_mat

def cluster_entropy(cluster):
    '''
    Compute cluster entropy

    Args:
        cluster: Cluster Dataframe

    Returns:
        entropy
    '''
    ent = 0
    for col in cluster.columns:
        distr = np.unique(cluster[col], return_counts=True)[1]/cluster.shape[0]
        ent += entropy(distr)
    return ent


def get_category(data, attr):
    '''
    Return category within the attribute with minimum entropy

    Args: 
        data: Dataframe
        attr: Attribute name to investigate 

    Returns:
        Category value
    '''
    categories = np.unique(data[attr])
    entropies = []
    for category in categories:
        ent = 0
        cluster = data.loc[data[attr] == category]

        for col in cluster.columns:
            distr = np.unique(cluster[col], return_counts=True)[
                1]/cluster.shape[0]
            ent += entropy(distr)
        value = ent#*cluster.shape[0]
        entropies.append(value)
    return categories[np.argmin(entropies)], np.min(entropies)


def get_category_max(data, attr):
    categories, counts = np.unique(data[attr], return_counts=True)
    return categories[counts == counts.max()][0]

def MIX(data):
    '''
    Returns attribute with highest mutual information with other attributes - Mutual Information indeX

    Attr:
        data: Dataframe

    Returns:
        Attribute index in the dataframe (int)
    '''
    a = dataset_information_measure(data)
    A = []
    for i in data.columns:
        A.append(np.unique(data[i]).shape[0])
    
    dia = np.diagonal(a)
    best_attr = np.argmax((a.sum(axis=1)-dia)/A)
    return best_attr

def clustering(data, K=30, verbose=0, vector=0, thresh=0):
    data = data.reset_index(drop=True)
    labels = np.zeros(shape=(data.shape[0],))
    clusters = []
    cluster_number = 0
    categs = []
    attributes = []

    while data.shape[0] and cluster_number < K-1:
        if verbose:
            print("Epoch# {} ".format(cluster_number))
        attr = MIX(data)
        attributes.append(data.columns[attr])
        cat, entr = get_category(data, data.columns[attr])
        M_data = data.drop([data.columns[attr]], axis =1)
        
        idx = get_cluster_idx(M_data, data.iloc[:, attr], cat, vector=vector, thresh=thresh)
        # print('\t Fraction of obj choosen{}/{}'.format(M_data.shape[0], len(idx)))
        if idx:          
            clus = data.loc[idx, :]
            bad = data.index.isin(idx)
            data = data[~bad]
        else:
            clus = data
            data = np.array([])
            
        categs.append(cat)
        clusters.append(clus) # list of clusters
        labels[clus.index] = cluster_number  # object cluster labels
        
        cluster_number += 1
        if cluster_number == K-1:
            if verbose:
                print('Epoch# {}'.format(cluster_number))
            clus = data
            clusters.append(clus)
            labels[clus.index] = cluster_number
            data = np.array([])
    return clusters, labels, attributes, categs


def clustering_refinement(data, labels, attributes, categories):
    # list of clusters from labels 
    clusters = []
    for c_idx in np.unique(labels):
        clusters.append(data.loc[labels == c_idx])
    
    sub_labels = dict()
    sub_attrs = dict()
    N_cluster = np.unique(labels).shape[0]
    for label in tqdm(range(N_cluster)): # labels are ordered as list(range(#clusters))
        clus = data.loc[labels == label]
        _, labls, attrs, categs = utils.clustering(clus, verbose=0)
        sub_labels[label] = labls
        try:
            strr = '{}:{}'.format(attributes[label], categories[label])
            sub_attrs[strr] = ['{}:{}'.format(i, j) for i, j in zip(attrs, categs)]
        except:
            sub_attrs['no_dominant_attr'] = ['{}:{}'.format(i, j) for i, j in zip(attrs, categs)]
            
    return sub_labels, sub_attrs

def plot_information_measure(info_mat, labels, ax=None, cbar_kw={}, **kwargs):
    if not ax:
        ax = plt.gca()

    im = ax.imshow(info_mat, **kwargs)

    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.xaxis.tick_top()

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # ax.xaxis.set_label_position('top')

    # plt.setp(ax.get_xticklabels(), rotation=45)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(info_mat.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(info_mat.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def plot_contingency_matrix(info_mat, lab_pred, lab_true, ax=None, cbar_kw={}, **kwargs):
    if not ax:
        ax = plt.gca()

    im = ax.imshow(info_mat, **kwargs)

    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(lab_true)))
    ax.set_yticks(np.arange(len(lab_pred)))
    ax.xaxis.tick_top()

    ax.set_xticklabels(lab_true)
    ax.set_yticklabels(lab_pred)
    # ax.xaxis.set_label_position('top')

    # plt.setp(ax.get_xticklabels(), rotation=45)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(info_mat.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(info_mat.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
