import numpy as np

def entropy(y):
    if len(y) == 0:
        return 0.0

    # Improve this:
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

# gain_ratio(parent, left, right)
#
# split_dataset(X, y, feature, threshold)
#
# majority_class(y)
#
# should_stop(y, depth, max_depth, min_samples)

y_left = np.array([0, 1])
y_right = np.array([1, 1, 1, 1, 1, 0, 0, 1])

si = split_information(y_left, y_right)
print(si)