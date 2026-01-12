import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.c45 import build_tree
from src.predict import predict
from src.predict_one import predict_one

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

tree = build_tree(
    X_train,
    y_train,
    max_depth=4,
    min_samples=2
)

y_pred = predict(X_test, tree)

print("Predictions:", y_pred[:20])
print("True labels:", y_test[:20])

acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", acc)

def print_tree(node, depth=0):
    indent = "  " * depth
    if node.is_leaf_node():
        print(f"{indent}Leaf → class {node.value}")
    else:
        print(f"{indent}X[{node.feature}] <= {node.threshold:.3f}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

print_tree(tree)
