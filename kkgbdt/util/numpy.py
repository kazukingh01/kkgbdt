import numpy as np


__all__ = [
    "softmax",
    "sigmoid",
    "rmse",
    "mae",
]


def softmax(x):
    """
    The code without njit is faster than with njit
    """
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_exp_x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rmse(x: np.ndarray, t: np.ndarray):
    return (x - t) ** 2 / 2.

def mae(x: np.ndarray, t: np.ndarray):
    return np.abs(x - t)
