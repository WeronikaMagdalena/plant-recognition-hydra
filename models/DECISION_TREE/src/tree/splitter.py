import numpy as np

def split_dataset(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def majority_class(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]
