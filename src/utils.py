import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,  # DEBUG for full trace
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def entropy(y):
    if len(y) == 0:
        logging.debug("Entropy called with empty array")
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()

    probabilities = probabilities[probabilities > 0]
    ent = -np.sum(probabilities * np.log2(probabilities))

    logging.debug(f"Entropy computed: {ent}")
    return ent


def information_gain(parent, left, right):
    """
    parent, left (child), right (child) - arrays of class labels
    """
    n_parent = len(parent)

    if n_parent == 0 or len(left) == 0 or len(right) == 0:
        logging.debug("Invalid split detected in information_gain")
        return 0.0

    parent_entropy = entropy(parent)

    weight_left = len(left) / n_parent
    weight_right = len(right) / n_parent

    children_entropy = (
        weight_left * entropy(left) +
        weight_right * entropy(right)
    )

    ig = parent_entropy - children_entropy
    logging.debug(f"Information Gain: {ig}")

    return ig


def split_information(left, right):
    n_left = len(left)
    n_right = len(right)
    n_total = n_left + n_right

    if n_total == 0:
        logging.debug("Split information called with empty split")
        return 0.0

    p_left = n_left / n_total
    p_right = n_right / n_total

    si = -sum(
        p * np.log2(p)
        for p in (p_left, p_right)
        if p > 0
    )

    logging.debug(f"Split Information: {si}")
    return si


def gain_ratio(parent, left, right):
    ig = information_gain(parent, left, right)
    si = split_information(left, right)

    if si == 0.0:
        logging.debug("Gain ratio undefined (split info = 0)")
        return 0.0

    gr = ig / si
    logging.debug(f"Gain Ratio: {gr}")

    return gr


def split_dataset(X, y, feature, threshold):
    logging.debug(
        f"Splitting dataset on feature {feature} with threshold {threshold}"
    )

    left_indices = X[:, feature] <= threshold
    right_indices = X[:, feature] > threshold

    X_left = X[left_indices]
    y_left = y[left_indices]
    X_right = X[right_indices]
    y_right = y[right_indices]

    logging.debug(
        f"Split result — Left: {len(y_left)}, Right: {len(y_right)}"
    )

    return X_left, y_left, X_right, y_right


def majority_class(y):
    if len(y) == 0:
        logging.debug("majority_class called with empty labels")
        return None

    values, counts = np.unique(y, return_counts=True)
    majority_index = np.argmax(counts)
    majority = values[majority_index]

    logging.debug(f"Majority class: {majority}")
    return majority


def should_stop(y, depth, max_depth, min_samples):
    if len(np.unique(y)) == 1:
        logging.info("Stopping: pure node")
        return True

    if depth >= max_depth:
        logging.info("Stopping: max depth reached")
        return True

    if len(y) < min_samples:
        logging.info("Stopping: not enough samples")
        return True

    return False
