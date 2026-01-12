import numpy as np
from src import utils


def find_best_split(X, y, n_bins=8, max_evals=5000):
    n_samples, n_features = X.shape

    best_gain_ratio = -1.0
    best_feature = None
    best_threshold = None

    # --- feature subsampling (huge speedup) ---
    n_sub_features = max(1, int(np.sqrt(n_features)))
    feature_indices = np.random.choice(
        n_features, n_sub_features, replace=False
    )

    evals = 0

    for feature_index in feature_indices:
        col = X[:, feature_index]

        # --- quantile-based thresholds ---
        thresholds = np.unique(
            np.percentile(col, np.linspace(0, 100, n_bins))
        )

        for threshold in thresholds:
            evals += 1
            if evals > max_evals:
                return best_feature, best_threshold

            left_mask = col <= threshold
            right_mask = ~left_mask

            if not left_mask.any() or not right_mask.any():
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            gr = utils.gain_ratio(y, y_left, y_right)

            if gr > best_gain_ratio:
                best_gain_ratio = gr
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold
