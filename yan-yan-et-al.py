import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# We define v = [w \gamma]
def dot_sigmoid(x, v):
    return sigmoid(np.dot(x, v[:-1]) + v[-1])


def yit_estimate(yit, xi, v):
    return 1 - dot_sigmoid(xi, v) if yit == 0 else dot_sigmoid(xi, v)


def yit_est_gauss(yit, xi, v):
    sigma = dot_sigmoid(xi, v)
    const = 1 / (sigma * np.sqrt(2 * np.pi))
    return (
        const * np.exp(1 / sigma ^ 2)
        if y == 1
        else const * np.exp(-1 / (2 * sigma ^ 2))
    )


def calc_p_tilde(xi, yi, v, a):
    p = 1
    z_factor = dot_sigmoid(xi, a)
    for t in range(len(yi)):
        yit = yi[t]
        p *= yit_estimate(yit, xi, v) * z_factor
    return p


# should we change this ^ to be flexible to bernoulli or gaussian?


def fit_classifier():
    pass


def fit_annotator():
    pass


def EM(x, y):
    N, D = x.shape
    N, T = y.shape
    p_tilde = np.zeros(N)
    v = np.zeros(T, D + 1)
    a = np.zeros(D + 1)
    while True:
        for i in range(N):
            xi = x[i, :]
            yi = y[i, :]
            p_tilde[i] = calc_p_tilde(xi, yi, v, a)
