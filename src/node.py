class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Index of the feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Class label if leaf node

    def is_leaf_node(self):
        return self.value is not None