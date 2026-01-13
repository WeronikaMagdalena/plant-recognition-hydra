import numpy as np
import pandas as pd

from tree_builder import build_tree
from predict import predict

# ============================================================
# CONFIGURATION
# ============================================================
N_CLASSES_TO_USE = 3        # None = all
MAX_DEPTH_CANDIDATES = [4, 6, 8]
MIN_SAMPLES_CANDIDATES = [10, 20]
RANDOM_STATE = 42

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv("../embeddings/balanced.csv")

# Optional class subsampling
if N_CLASSES_TO_USE is not None:
    class_counts = df["label"].value_counts()
    selected_classes = class_counts.index[:N_CLASSES_TO_USE]
    df = df[df["label"].isin(selected_classes)]

print("Using classes:", df["label"].nunique())

X = df.drop(columns=["label"]).values.astype(float)
y = df["label"].values
classes = np.unique(y)
n_classes = len(classes)

# ============================================================
# Train / Val / Test split (70 / 15 / 15)
# ============================================================
rng = np.random.default_rng(RANDOM_STATE)
indices = rng.permutation(len(y))

n_total = len(y)
n_train = int(0.7 * n_total)
n_val   = int(0.15 * n_total)

train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val     = X[val_idx], y[val_idx]
X_test, y_test   = X[test_idx], y[test_idx]

print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

# ============================================================
# Hyperparameter selection (VALIDATION)
# ============================================================
best_score = -1
best_params = None

for max_depth in MAX_DEPTH_CANDIDATES:
    for min_samples in MIN_SAMPLES_CANDIDATES:

        tree = build_tree(
            X_train,
            y_train,
            max_depth=max_depth,
            min_samples=min_samples
        )

        y_val_pred = predict(X_val, tree)
        val_accuracy = np.mean(y_val_pred == y_val)

        print(
            f"Val accuracy (depth={max_depth}, min_samples={min_samples}): "
            f"{val_accuracy:.4f}"
        )

        if val_accuracy > best_score:
            best_score = val_accuracy
            best_params = (max_depth, min_samples)

print("\nBest validation params:", best_params)
print("Best validation accuracy:", best_score)

# ============================================================
# Retrain on Train + Validation
# ============================================================
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

tree = build_tree(
    X_train_full,
    y_train_full,
    max_depth=best_params[0],
    min_samples=best_params[1]
)

# ============================================================
# Final evaluation on TEST
# ============================================================
y_test_pred = predict(X_test, tree)
test_accuracy = np.mean(y_test_pred == y_test)

print(f"\nFINAL TEST ACCURACY: {test_accuracy:.4f}")

# ============================================================
# Confusion Matrix
# ============================================================
label_to_index = {label: i for i, label in enumerate(classes)}
conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

for true, pred in zip(y_test, y_test_pred):
    conf_matrix[label_to_index[true], label_to_index[pred]] += 1

print("\nConfusion matrix shape:", conf_matrix.shape)
print(conf_matrix[:min(5, n_classes), :min(5, n_classes)])

# ============================================================
# Tree statistics
# ============================================================
def count_nodes(node):
    if node is None:
        return 0
    return count_nodes(node.left) + count_nodes(node.right)

def tree_depth(node):
    if node is None or node.is_leaf_node():
        return 1
    return max(tree_depth(node.left), tree_depth(node.right))

print("\nTree statistics:")
print("Total nodes:", count_nodes(tree))
print("Max depth :", tree_depth(tree) - 1)
