import numpy as np

def predict_one(x, node):
    while not node.is_leaf_node():
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])
