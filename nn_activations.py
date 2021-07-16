import jax.numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def apr1(x):
    return np.float32(1) - np.float32(1) / (x + np.float32(1))
