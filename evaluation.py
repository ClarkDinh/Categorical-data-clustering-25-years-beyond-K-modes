import numpy as np
from scipy.stats import mode
from scipy.special import comb
from math import log
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def accuracy(predicted_row_labels, true_row_labels):
    """Get the best accuracy.

    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information.
    predicted_row_labels: array-like
        The row labels predicted by the model.

    Returns
    -------
    float
        Best value of accuracy.
    """

    true_row_labels = grp2idx(true_row_labels)  # Convert to integer labels
    predicted_row_labels = grp2idx(predicted_row_labels)  # Convert to integer labels
    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    row_ind, col_ind = linear_sum_assignment(_make_cost_m(cm))
    total = cm[row_ind, col_ind].sum()

    return total / np.sum(cm)

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

# Normalized Mutual Information: Normalizes mutual information by entropy.
# Higher is better.
def nmi(x, y):
    N = x.size
    I = mutual_info(x, y)
    Hx = 0
    for l1 in np.unique(x):
        l1_count = np.where(x == l1)[0].size
        Hx += -(l1_count / N) * log(l1_count / N)
    Hy = 0
    for l2 in np.unique(y):
        l2_count = np.where(y == l2)[0].size
        Hy += -(l2_count / N) * log(l2_count / N)
    return I / ((Hx + Hy) / 2)

# Adjusted Rand Index: Measures the similarity of predicted and actual labels.
# Higher is better.
def rand(predicted, labels):
    predicted = grp2idx(predicted)
    labels = grp2idx(labels)
    a = np.zeros((len(np.unique(labels)), len(np.unique(predicted))))
    for i in range(a.shape[0]):
        for i2 in range(a.shape[1]):
            a[i, i2] = np.intersect1d(np.where(labels == i)[0], np.where(predicted == i2)[0]).size
    cij = 0
    a = a.astype(float)
    for i in range(a.shape[0]):
        for i2 in range(a.shape[1]):
            if a[i, i2] > 1:
                cij += comb(a[i, i2], 2, exact=True)
    ci = 0
    for i in range(a.shape[0]):
        if np.sum(a[i, :]) > 1:
            ci += comb(np.sum(a[i, :]), 2, exact=True)
    cj = 0
    for i in range(a.shape[1]):
        if np.sum(a[:, i]) > 1:
            cj += comb(np.sum(a[:, i]), 2, exact=True)
    cn = comb(len(labels), 2, exact=True)
    nominator = cij - ((ci * cj) / cn)
    denominator = 0.5 * (ci + cj) - (ci * cj / cn)
    return nominator / denominator

# Purity: Measures the quality of clustering based on agreement with class labels.
# Higher is better.
def purity(predicted, label):
    total_max = 0
    unique_predicted = np.unique(predicted)
    for x in unique_predicted:
        max_count = 0
        count = {}
        for i in range(len(predicted)):
            if predicted[i] != x:
                continue
            count[label[i]] = count.get(label[i], 0) + 1
            if max_count < count[label[i]]:
                max_count = count[label[i]]
        total_max += max_count
    return total_max / len(predicted)

# Mutual Information: Measures the mutual dependence between predicted and actual labels.
# Higher is better.
def mutual_info(x, y):
    N = x.size
    I = 0.0
    eps = np.finfo(float).eps
    for l1 in np.unique(x):
        for l2 in np.unique(y):
            l1_ids = np.where(x == l1)[0]
            l2_ids = np.where(y == l2)[0]
            pxy = (np.intersect1d(l1_ids, l2_ids).size / N) + eps
            I += pxy * log(pxy / ((l1_ids.size / N) * (l2_ids.size / N)))
    return I

# Convert labels to indices.
def grp2idx(labels):
    unique_labels = np.unique(labels)
    inds = {label: idx for idx, label in enumerate(unique_labels)}
    return np.array([inds[label] for label in labels])

if __name__ == "__main__":
    # Example usage
    predicted_labels = np.array([1, 1, 1, 2, 2, 2, 3, 4])
    true_labels = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'])

    # Calculate and print metrics
    print("Accuracy:", accuracy(predicted_labels, true_labels))
    print("Purity:", purity(predicted_labels, true_labels))
    print("Normalized Mutual Information:", nmi(predicted_labels, true_labels))
    print("Adjusted Rand Index:", rand(predicted_labels, true_labels))
