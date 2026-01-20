import numpy as np
from .base_svm import BaseSVM

class KernelSVM(BaseSVM):
    def __init__(self, lr=0.01, lmb=0.01, epochs=30, kernel='rbf', gamma=0.5):
        super().__init__(lr, lmb, epochs)
        self.kernel = kernel
        self.gamma = gamma
        self.X_train = None
        self.y_train = None

    def _rbf_kernel(self, X1, X2):
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist)

    def _poly_kernel(self, X1, X2, deg=3):
        return (self.gamma * np.dot(X1, X2.T) + 1) ** deg

    def _compute_kernel(self, X1, X2):
        if self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        return self._poly_kernel(X1, X2)

    def _find_nearest(self, x, X):
        dists = np.sum((X - x) ** 2, axis=1)
        return np.argmin(dists)

    def _update_alphas(self, alphas, idx, y_bin):
        alphas *= (1 - self.lr * self.lmb)
        alphas = np.clip(alphas, -100, 100)
        alphas[idx] += self.lr * y_bin
        return alphas

    def _update_kernel_weights(self, i, alphas, b, y_bin, margin):
        if margin < 1:
            nearest = self._find_nearest(self.X_train[i], self.X_train)
            alphas = self._update_alphas(alphas, nearest, y_bin[i])
            b += self.lr * y_bin[i]
            b = np.clip(b, -100, 100)
        else:
            alphas *= (1 - self.lr * self.lmb)
        return alphas, b

    def _train_epoch(self, y_bin, alphas, b):
        n = len(self.X_train)
        idx = np.random.permutation(n)
        for i in idx:
            K_i = self._compute_kernel(self.X_train[i:i+1], self.X_train).flatten()
            decision = np.dot(K_i, alphas * y_bin) + b
            margin = y_bin[i] * decision
            alphas, b = self._update_kernel_weights(i, alphas, b, y_bin, margin)
        return alphas, b

    def _train_one_class(self, X, y, c):
        y_bin = self._binary_labels(y, c)
        alphas = np.zeros(len(X))
        b = 0

        for _ in range(self.epochs):
            alphas, b = self._train_epoch(y_bin, alphas, b)

        return {'alphas': alphas, 'b': b}

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        super().fit(X, y)

    def _compute_scores(self, X):
        scores = np.zeros((len(X), self.n_cls))
        for c in range(self.n_cls):
            alphas = self.clfs[c]['alphas']
            b = self.clfs[c]['b']
            K = self._compute_kernel(X, self.X_train)
            y_bin = self._binary_labels(self.y_train, c)
            scores[:, c] = np.dot(K, alphas * y_bin) + b
        return scores
