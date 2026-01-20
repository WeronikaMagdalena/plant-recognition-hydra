import numpy as np
from abc import ABC, abstractmethod

class BaseSVM(ABC):
    def __init__(self, lr=0.01, lmb=0.01, epochs=50):
        self.lr = lr
        self.lmb = lmb
        self.epochs = epochs
        self.clfs = {}
        self.n_cls = None

    def _binary_labels(self, y, c):
        return np.where(y == c, 1, -1)

    def _init_weights(self, n_feat):
        w = np.zeros(n_feat)
        b = 0
        return w, b

    def fit(self, X, y):
        self.n_cls = int(y.max() + 1)
        for c in range(self.n_cls):
            self.clfs[c] = self._train_one_class(X, y, c)

    def predict(self, X):
        scores = self._compute_scores(X)
        return np.argmax(scores, axis=1)

    @abstractmethod
    def _train_one_class(self, X, y, c):
        pass

    @abstractmethod
    def _compute_scores(self, X):
        pass
