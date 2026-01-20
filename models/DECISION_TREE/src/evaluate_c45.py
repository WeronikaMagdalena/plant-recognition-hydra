import logging
import pandas as pd

from config import *
from tree.builder import build_tree
from models.DECISION_TREE.src.tree.predictor import predict
from metrics import *
from stats import *
from tree.post_prunning import post_prune

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

df = pd.read_csv("../embeddings/balanced.csv")

if N_CLASSES_TO_USE is not None:
    class_counts = df["label"].value_counts()
    selected_classes = class_counts.index[:N_CLASSES_TO_USE]
    df = df[df["label"].isin(selected_classes)]

X = df.drop(columns=["label"]).values.astype(float)
y = df["label"].values
classes = np.unique(y)
n_classes = len(classes)

rng = np.random.default_rng(RANDOM_STATE)
indices = rng.permutation(len(y))

n_total = len(y)
n_train = int(TRAIN_RATIO * n_total)
n_val = int(VAL_RATIO * n_total)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

best_score = -1
best_params = None

for max_depth in MAX_DEPTH_CANDIDATES:
    for min_samples in MIN_SAMPLES_CANDIDATES:
        tree = build_tree(
            X_train, y_train,
            max_depth=max_depth,
            min_samples=min_samples,
            n_bins=N_BINS,
            max_evals=MAX_EVALS,
            random_state=RANDOM_STATE
        )
        y_val_pred = predict(X_val, tree)
        val_acc = accuracy(y_val, y_val_pred)
        logger.info(f"Val accuracy (depth={max_depth}, min_samples={min_samples}): {val_acc:.4f}")
        if val_acc > best_score:
            best_score = val_acc
            best_params = (max_depth, min_samples)

X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

tree = build_tree(
    X_train_full, y_train_full,
    max_depth=best_params[0],
    min_samples=best_params[1],
    n_bins=N_BINS,
    max_evals=MAX_EVALS,
    random_state=RANDOM_STATE
)

tree = post_prune(tree, X_val, y_val)

y_test_pred = predict(X_test, tree)
test_acc = accuracy(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

print(f"FINAL TEST ACCURACY: {test_acc:.4f}")
print(f"Confusion matrix shape: {cm.shape}")
print(cm[:min(5, n_classes), :min(5, n_classes)])
print("Tree statistics:")
print("Total nodes:", count_nodes(tree))
print("Max depth :", tree_depth(tree) - 1)