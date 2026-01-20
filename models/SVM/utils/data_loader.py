import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, test_size=0.1, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def load(self, data_type="balanced"):
        filename = self._get_filename(data_type)
        X, y = self._read_csv(filename)
        return self._split_data(X, y)

    def _get_filename(self, data_type):
        return "../balanced.csv" if data_type == "balanced" else "../unbalanced.csv"

    def _read_csv(self, filename):
        df = pd.read_csv(filename)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        return X, y

    def _split_data(self, X, y):
        np.random.seed(self.random_state)
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)
        indices = np.random.permutation(n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def _get_stratify(self, y):
        _, counts = np.unique(y, return_counts=True)
        return y if np.all(counts >= 2) else None
