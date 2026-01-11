import numpy as np

def entropy(y):
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()

    probabilities = probabilities[probabilities > 0]

    return -np.sum(probabilities * np.log2(probabilities))

# information_gain(parent, left, right)
#
# split_information(left, right)
#
# gain_ratio(parent, left, right)
#
# split_dataset(X, y, feature, threshold)
#
# majority_class(y)
#
# should_stop(y, depth, max_depth, min_samples)