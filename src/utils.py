import numpy as np

def entropy(y):
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()

    probabilities = probabilities[probabilities > 0]

    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(parent, left, right):
    """
    parent, left (child), right (child) - arrays of class labels
    """
    n_parent = len(parent)

    # Invalid split
    if n_parent == 0 or len(left) == 0 or len(right) == 0:
        return 0.0

    parent_entropy = entropy(parent)

    weight_left = len(left) / n_parent
    weight_right = len(right) / n_parent

    children_entropy = (
            weight_left * entropy(left) +
            weight_right * entropy(right)
    )

    return parent_entropy - children_entropy


def split_information(left, right):
    n_left = len(left)
    n_right = len(right)
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    p_left = n_left / n_total
    p_right = n_right / n_total

    return - (p_left * np.log2(p_left) + p_right * np.log2(p_right))


def gain_ratio(parent, left, right):
    ig = information_gain(parent, left, right)
    si = split_information(left, right)

    if si == 0.0:
        return 0.0

    return ig / si


def split_dataset(X, y, feature, threshold):
    # arrays of trues and falses to deterimine by index
    left_indices = X[:, feature] <= threshold
    right_indices = X[:, feature] > threshold

    # back to whole embeddings (X) and labels (y)
    X_left = X[left_indices]
    y_left = y[left_indices]
    X_right = X[right_indices]
    y_right = y[right_indices]

    return X_left, y_left, X_right, y_right


def majority_class(y):
    if len(y) == 0:
        return None
    values, counts = np.unique(y, return_counts=True)
    majority_index = np.argmax(counts)
    return values[majority_index]


def should_stop(y, depth, max_depth, min_samples):
    # Stop if all labels are the same
    if len(np.unique(y)) == 1:
        return True

    if depth >= max_depth:
        return True

    # Stop if not enough samples to split
    if len(y) < min_samples:
        return True

    return False
