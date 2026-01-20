import numpy as np

class DataUtils:
    @staticmethod
    def evaluate(model, X, y):
        pred = model.predict(X)
        return np.mean(pred == y)

    @staticmethod
    def shuffle_data(X, y, seed=42):
        idx = np.arange(len(X))
        np.random.seed(seed)
        np.random.shuffle(idx)
        return X[idx], y[idx]

    @staticmethod
    def split_data(X, y, train_size=100):
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        return X_train, y_train, X_test, y_test
