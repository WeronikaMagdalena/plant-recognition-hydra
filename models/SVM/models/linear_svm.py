import numpy as np
from .base_svm import BaseSVM

class LinearSVM(BaseSVM):
    def _compute_margin(self, x, y_bin, w, b):
        return y_bin * (np.dot(x, w) - b)

    def _update_weights(self, x, y_bin, w, b, margin):
        if margin < 1:
            w = w - self.lr * self.lmb * w + self.lr * y_bin * x
            b = b + self.lr * y_bin
        else:
            w = w - self.lr * self.lmb * w
        return w, b

    def _train_epoch(self, X, y_bin, w, b):
        n = len(X)
        idx = np.random.permutation(n)
        for i in idx:
            margin = self._compute_margin(X[i], y_bin[i], w, b)
            w, b = self._update_weights(X[i], y_bin[i], w, b, margin)
        return w, b

    def _train_one_class(self, X, y, c):
        y_bin = self._binary_labels(y, c)
        w, b = self._init_weights(X.shape[1])

        for _ in range(self.epochs):
            w, b = self._train_epoch(X, y_bin, w, b)

        return {'w': w, 'b': b}

    def _compute_scores(self, X):
        scores = np.zeros((len(X), self.n_cls))
        for c in range(self.n_cls):
            w = self.clfs[c]['w']
            b = self.clfs[c]['b']
            scores[:, c] = np.dot(X, w) - b
        return scores
