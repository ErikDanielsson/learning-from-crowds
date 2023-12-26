import numpy as np
from utils import dot_sigmoid
import scipy.optimize

e = 1e-9


def s_log_loss(sigma, y):
    return y * np.log(sigma + e) + (1 - y) * np.log(1 - sigma + e)


vec_log_loss = np.vectorize(s_log_loss)


def log_loss(w, y, x, l):
    s = dot_sigmoid(x, w)
    return -sum(vec_log_loss(s, y)) + l * np.dot(w, w)


def gradient_log_loss(w, y, x, l):
    s = dot_sigmoid(x, w)
    return -np.dot(x.T, y - s) + l * w


def hessian_log_loss(w, y, x, l):
    s = dot_sigmoid(x, w)
    S = np.diag(np.multiply(s, 1 - s))
    return np.linalg.multi_dot((x.T, S, x)) + l * np.eye(len(w))


def log_reg(y, x, l, D):
    return scipy.optimize.minimize(
        log_loss,
        np.random.randn(D + 1),
        jac=gradient_log_loss,
        hess=hessian_log_loss,
        args=(y, x, l),
        method="trust-exact",
        options={"maxiter": 20000},
    )
