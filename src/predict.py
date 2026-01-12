import numpy as np
from src.predict_one import predict_one


def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])
