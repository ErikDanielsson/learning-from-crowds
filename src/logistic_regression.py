import numpy as np
from utils import dot_sigmoid
import scipy.optimize


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


def log_reg(y, x, sigma, D):
    return scipy.optimize.minimize(
        log_loss,
        np.random.randn(D + 1),
        jac=gradient_log_loss,
        hess=hessian_log_loss,
        args=(y, x, sigma),
        method="trust-krylov",
        options={"maxiter": 20000},
    )
