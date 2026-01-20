def count_nodes(node):
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)

def tree_depth(node):
    if node is None or node.is_leaf_node():
        return 1
    return 1 + max(tree_depth(node.left), tree_depth(node.right))
