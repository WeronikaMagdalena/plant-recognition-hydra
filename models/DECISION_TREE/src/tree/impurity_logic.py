import numpy as np

def entropy(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

# input - arrays of class labels
def information_gain(parent, left, right):
    n_parent = len(parent)
    if n_parent == 0 or len(left) == 0 or len(right) == 0:
        return 0.0
    parent_entropy = entropy(parent)
    weight_left = len(left) / n_parent
    weight_right = len(right) / n_parent
    children_entropy = weight_left * entropy(left) + weight_right * entropy(right)
    return parent_entropy - children_entropy

def split_information(left, right):
    n_left = len(left)
    n_right = len(right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    p_left = n_left / n_total
    p_right = n_right / n_total
    return -sum(p * np.log2(p) for p in (p_left, p_right) if p > 0)

def gain_ratio(parent, left, right):
    ig = information_gain(parent, left, right)
    si = split_information(left, right)
    if si == 0.0:
        return 0.0
    return ig / si
