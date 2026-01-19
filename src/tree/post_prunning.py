from src.tree.predictor import predict
from src.tree.splitter import majority_class
from src.tree.node import Node
import numpy as np

def post_prune(tree, X_val, y_val):

    def prune_node(node):
        if node.is_leaf_node():
            return node

        node.left = prune_node(node.left)
        node.right = prune_node(node.right)

        if node.left.is_leaf_node() and node.right.is_leaf_node():
            y_pred_subtree = predict(X_val, node)
            acc_subtree = (y_pred_subtree == y_val).mean()

            candidate_value = majority_class(np.array([node.left.value, node.right.value]))
            leaf_node = Node(value=candidate_value)

            y_pred_leaf = predict(X_val, leaf_node)
            acc_leaf = (y_pred_leaf == y_val).mean()

            if acc_leaf > acc_subtree:
                return leaf_node

        return node

    return prune_node(tree)
