import numpy as np
import scipy.optimize

from utils import dot_sigmoid
from logistic_regression import log_reg


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
    w = log_reg(vote, X_1, 0, D).x
    return w


def true_classifier(X, y, sigma):
    N, D = X.shape
    X_1 = np.hstack((X, np.ones((N, 1))))
    w = log_reg(y, X_1, 0, D).x
    return w


def concat(X, y, sigma):
    N, D = X.shape
    N, n_experts = y.shape
    X_1 = np.hstack((X, np.ones((N, 1))))
    X = np.vstack([X_1 for _ in range(n_experts)])
    h = np.hstack([y[:, i] for i in range(n_experts)])
    w = log_reg(h, X, 0, D).x
    return w
