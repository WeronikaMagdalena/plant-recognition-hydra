import logging
import numpy as np
from models.DECISION_TREE.src.tree.node import Node
from models.DECISION_TREE.src.tree.best_split import find_best_split
from models.DECISION_TREE.src.tree.splitter import split_dataset, majority_class
from models.DECISION_TREE.src.tree.pre_prunning import should_stop

logger = logging.getLogger(__name__)

def build_tree(X, y, depth=0, max_depth=10, min_samples=2, n_bins=8, max_evals=5000, random_state=None):
    logger.info(f"{'  ' * depth}Building node at depth {depth} (samples={len(y)})")

    if should_stop(y, depth, max_depth, min_samples):
        return Node(value=majority_class(y))

    rng = np.random.default_rng(random_state)
    feature, threshold = find_best_split(X, y, n_bins=n_bins, max_evals=max_evals, rng=rng)

    if feature is None:
        return Node(value=majority_class(y))

    X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)

    left_child = build_tree(X_left, y_left, depth + 1, max_depth, min_samples,
                            n_bins=n_bins, max_evals=max_evals, random_state=random_state)

    right_child = build_tree(X_right, y_right, depth + 1, max_depth, min_samples,
                             n_bins=n_bins, max_evals=max_evals, random_state=random_state)

    return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)