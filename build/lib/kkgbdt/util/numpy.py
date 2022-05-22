import numpy as np


__all__ = [
    "softmax",
    "sigmoid",
    "rmse",
    "mae",
]


def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
    return f

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rmse(x: np.ndarray, t: np.ndarray):
    return (x - t) ** 2 / 2.

def mae(x: np.ndarray, t: np.ndarray):
    return np.abs(x - t)
