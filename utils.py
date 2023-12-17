import numpy as np


def generate_data(N, w, n=2, minc=0, maxc=1):
    x1 = (maxc - minc) * np.random.random((N, n)) + minc
    ones = np.ones((N, 1))
    x = np.concatenate((x1, ones), axis=1)
    y = 1 * (np.dot(x, w) > 0)
    return x, y


def expert_advice(y, x, w):
    N = len(y)
    M = w.shape[0]
    x = np.hstack((x, np.ones((N, 1))))
    advice = np.zeros((N, M))
    for i, yi in enumerate(y):
        if yi == 1:
            for j in range(M):
                advice[i, j] = 1 * (
                    np.random.uniform(0, 1) <= dot_sigmoid(x[i, :], w[j, :])
                )
        else:
            for j in range(M):
                advice[i, j] = 1 * (
                    np.random.uniform(0, 1) >= dot_sigmoid(x[i, :], w[j, :])
                )
    return advice


def sigmoid(x):
    if x < -500:
        return 0
    return 1 / (1 + np.exp(-x))


# We define v = [w \gamma]
def dot_sigmoid(x, v):
    return sigmoid(np.dot(x, v))
