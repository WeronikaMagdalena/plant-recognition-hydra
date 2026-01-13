import logging
from best_split import find_best_split
from node import Node
from utils import should_stop, majority_class, split_dataset

logger = logging.getLogger(__name__)

def build_tree(X, y, depth=0, max_depth=10, min_samples=2):
    logger.info(
        f"{'  ' * depth}Building node at depth {depth} "
        f"(samples={len(y)})"
    )

    if should_stop(y, depth, max_depth, min_samples):
        leaf_value = majority_class(y)
        logger.info(
            f"{'  ' * depth}Creating leaf node "
            f"with value={leaf_value}"
        )
        return Node(value=leaf_value)

    feature, threshold = find_best_split(X, y)

    if feature is None:
        leaf_value = majority_class(y)
        logger.warning(
            f"{'  ' * depth}No valid split found — "
            f"creating leaf with value={leaf_value}"
        )
        return Node(value=leaf_value)

    logger.info(
        f"{'  ' * depth}Best split: feature={feature}, "
        f"threshold={threshold}"
    )

    X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)

    logger.debug(
        f"{'  ' * depth}Split sizes — "
        f"left={len(y_left)}, right={len(y_right)}"
    )

    left_child = build_tree(
        X_left, y_left, depth + 1, max_depth, min_samples
    )
    right_child = build_tree(
        X_right, y_right, depth + 1, max_depth, min_samples
    )

    return Node(
        feature=feature,
        threshold=threshold,
        left=left_child,
        right=right_child
    )
