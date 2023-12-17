import numpy as np
import scipy.optimize
from utils import *
from multiprocessing import Pool


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
    N, D = x.shape
    N, T = y.shape
    p_tilde = np.random.rand(N)
    soft_label = np.zeros((N, T))
    v = np.zeros((T, D + 1))
    v[:, -1] = 10
    a = np.zeros(D + 1)
    a_new = np.random.randn(D + 1)
    l_prev = -np.inf
    l_curr = 0
    ls = []

    x_1 = np.hstack((x, np.ones((N, 1))))
    while np.linalg.norm(a - a_new) > epsilon_tot:
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
        res = scipy.optimize.minimize(
            log_loss,
            np.random.randn(D + 1),
            jac=gradient_log_loss,
            hess=hessian_log_loss,
            args=(p_tilde, x_1, sigma),
            method="trust-exact",
        )
        print(np.arccos(np.dot(a_new, a) / np.linalg.norm(a_new) / np.linalg.norm(a)))
        a_new = res.x
        a_new /= a_new[0]
        if res.success:
            pass
        else:
            print("fail a")
            print(res)

        for t in range(T):
            res = scipy.optimize.minimize(
                log_loss,
                np.random.randn(D + 1),
                jac=gradient_log_loss,
                hess=hessian_log_loss,
                args=(soft_label[:, t], x_1, sigma),
                method="trust-exact",
            )
            v[t, :] = res.x
            if res.success:
                pass
            else:
                print("fail v")
                print(res)

        l_curr = real_likelihood(a_new, v, x_1, y, N, T)
        ls.append(l_curr)
        print(np.linalg.norm(a - a_new))
        # print(a_new)

    return a, v, ls
