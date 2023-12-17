import numpy as np
import scipy.optimize

from utils import dot_sigmoid


def s_log_loss(sigma, y):
    return y * np.log(sigma) + (1 - y) * np.log(1 - sigma)


vec_log_loss = np.vectorize(s_log_loss)


def log_loss(w, y, x, sigma):
    sigma = dot_sigmoid(x, w)
    return -sum(vec_log_loss(sigma, y))


def gradient_log_loss(w, y, x, sigma):
    s = dot_sigmoid(x, w)
    return -np.dot(x.T, y - s)


def hessian_log_loss(w, y, x, sigma):
    s = dot_sigmoid(x, w)
    S = np.diag(np.multiply(s, 1 - s))
    return np.linalg.multi_dot((x.T, S, x))


def majority_opinion(advice):
    return 1 * (np.mean(advice, axis=1) >= 0.5)


def majority(X, advice, sigma):
    N, D = X.shape
    X_1 = np.hstack((X, np.ones((N, 1))))
    vote = majority_opinion(advice)
    w = scipy.optimize.minimize(
        log_loss,
        np.random.rand(D + 1),
        args=(vote, X_1, sigma),
        jac=gradient_log_loss,
        hess=hessian_log_loss,
        method="trust-exact",
    ).x
    return w
