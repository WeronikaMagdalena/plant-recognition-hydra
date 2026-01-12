import numpy as np

def filter_and_balance(X, y, min_samples_per_class=50, random_state=42):
    """
    1️⃣ Remove classes with fewer than min_samples_per_class samples
    2️⃣ Balance remaining classes by undersampling to the smallest class
    """
    rng = np.random.default_rng(random_state)

    # Step 1: filter out small classes
    classes, counts = np.unique(y, return_counts=True)
    valid_classes = classes[counts >= min_samples_per_class]

    mask = np.isin(y, valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]

    # Step 2: balance remaining classes
    classes, counts = np.unique(y_filtered, return_counts=True)
    min_count = counts.min()

    X_balanced = []
    y_balanced = []

    for cls in classes:
        idx = np.where(y_filtered == cls)[0]
        selected = rng.choice(idx, size=min_count, replace=False)
        X_balanced.append(X_filtered[selected])
        y_balanced.append(y_filtered[selected])

    X_balanced = np.vstack(X_balanced)
    y_balanced = np.concatenate(y_balanced)

    # Shuffle
    perm = rng.permutation(len(y_balanced))
    return X_balanced[perm], y_balanced[perm]
