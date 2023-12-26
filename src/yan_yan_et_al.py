import numpy as np
import scipy.optimize
from utils import *
from logistic_regression import log_reg


def yit_estimate(yit, xi, v):
    return 1 - slow_sigmoid(xi, v) if yit == 0 else slow_sigmoid(xi, v)


def calc_p_tilde(xi, yi, v, a):
    p = 1
    q = 1
    z_factor = slow_sigmoid(xi, a)
    for t in range(len(yi)):
        e = yit_estimate(yi[t], xi, v[t, :])
        f = 1 - e
        p *= e
        q *= f
    return p * z_factor / (p * z_factor + q * (1 - z_factor))


def soft_lab(yit, p_tilde):
    return yit * p_tilde + (1 - yit) * (1 - p_tilde)


def real_likelihood(a, v, x, y, N, T):
    zf = dot_sigmoid(x, a)
    s = 0
    for i in range(N):
        xi = x[i, :]
        yi = y[i, :]
        for t in range(T):
            yit = yi[t]
            yf = yit_estimate(yit, xi, v[t, :])
            s += np.log(zf[i] * yf + (1 - zf[i]) * (1 - yf))
    return s


def posterior(x, y, a, v):
    T = v.shape[0]
    zf = dot_sigmoid(x, a)
    likelihood = 1
    not_likelihood = 1
    for t in range(T):
        yt = y[t]
        yf = yit_estimate(yt, x, v[t, :])
        likelihood *= yf
        not_likelihood *= 1 - yf
    return likelihood * zf / (likelihood * zf + not_likelihood * (1 - zf))


def yan_yan_et_al(x, y, epsilon_tot, sigma):
    print(f"YAN YAN {sigma}")
    N, D = x.shape
    N, T = y.shape
    p_tilde = np.random.rand(N)
    soft_label = np.zeros((N, T))
    v = np.zeros((T, D + 1))
    v[:, -1] = 10
    a = np.zeros(D + 1)
    a_new = np.random.randn(D + 1)
    l_prev = -1e11
    l_curr = -1e10
    ls = []

    x_1 = np.hstack((x, np.ones((N, 1))))
    while l_curr - l_prev > epsilon_tot:
        # print(f"diff: {l_curr - l_prev}")
        l_prev = l_curr
        a = a_new
        # E-step
        for i in range(N):
            xi = x_1[i, :]
            yi = y[i, :]
            p_tilde[i] = calc_p_tilde(xi, yi, v, a)

        for i in range(N):
            yi = y[i, :]
            p_ti = p_tilde[i]
            for t in range(T):
                soft_label[i, t] = soft_lab(yi[t], p_ti)

        # M-step
        res = log_reg(p_tilde, x_1, sigma, D)
        a_new = res.x
        for t in range(T):
            res = log_reg(soft_label[:, t], x_1, 0, D)
            v[t, :] = res.x

        l_curr = real_likelihood(a_new, v, x_1, y, N, T)
        ls.append(l_curr)
        # print(l_curr, np.linalg.norm(a_new))
        # print(a_new)
        # print(v)

    return a, v, ls
