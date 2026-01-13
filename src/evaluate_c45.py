import numpy as np
import pandas as pd

from tree_builder import build_tree
from predict import predict

# ============================================================
# 🔧 CONFIGURATION
# ============================================================
N_CLASSES_TO_USE = 2      # None = use all classes
MAX_DEPTH = 6
MIN_SAMPLES = 20
RANDOM_STATE = 42

# ============================================================
# 📥 Load dataset
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
# 🔀 Train / Test split
# ============================================================
rng = np.random.default_rng(RANDOM_STATE)
indices = rng.permutation(len(y))

split = int(0.8 * len(y))
train_idx = indices[:split]
test_idx  = indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test   = X[test_idx], y[test_idx]

# ============================================================
# 🌳 Train tree
# ============================================================
tree = build_tree(
    X_train,
    y_train,
    max_depth=MAX_DEPTH,
    min_samples=MIN_SAMPLES
)

# ============================================================
# 🔮 Predict
# ============================================================
y_pred = predict(X_test, tree)

accuracy = np.mean(y_pred == y_test)
print(f"\nOverall accuracy: {accuracy:.4f}")

# ============================================================
# 📊 Confusion Matrix (MANUAL)
# ============================================================
label_to_index = {label: i for i, label in enumerate(classes)}
conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

for true, pred in zip(y_test, y_pred):
    conf_matrix[label_to_index[true], label_to_index[pred]] += 1

print("\nConfusion matrix shape:", conf_matrix.shape)

# 🔍 Print small preview (top-left corner)
preview_size = min(5, n_classes)
print("\nConfusion matrix preview:")
print(conf_matrix[:preview_size, :preview_size])

# ============================================================
# 📈 Per-class accuracy
# ============================================================
per_class_acc = {}

for cls in classes:
    idx = label_to_index[cls]
    correct = conf_matrix[idx, idx]
    total = conf_matrix[idx].sum()
    per_class_acc[cls] = correct / total if total > 0 else 0.0

mean_class_acc = np.mean(list(per_class_acc.values()))
print("\nMean per-class accuracy:", mean_class_acc)

# ============================================================
# 📋 Per-class accuracy (Top 10 and Bottom 10)
# ============================================================
sorted_acc = sorted(
    per_class_acc.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 10 classes by accuracy:")
for cls, acc in sorted_acc[:10]:
    print(f"Class {cls}: {acc:.4f}")

print("\nBottom 10 classes by accuracy:")
for cls, acc in sorted_acc[-10:]:
    print(f"Class {cls}: {acc:.4f}")

# ============================================================
# 🌳 Tree statistics
# ============================================================
def count_nodes(node):
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)

def tree_depth(node):
    if node is None or node.is_leaf_node():
        return 1
    return 1 + max(tree_depth(node.left), tree_depth(node.right))

print("\nTree statistics:")
print("Total nodes:", count_nodes(tree))
print("Max depth :", tree_depth(tree))
