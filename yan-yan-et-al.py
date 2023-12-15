import numpy as np
import scipy.optimize
from utils import *
from multiprocessing import Pool
from sklearn import tree


def yit_estimate(yit, xi, v):
    return 1 - dot_sigmoid(xi, v) if yit == 0 else dot_sigmoid(xi, v)


def calc_p_tilde(xi, yi, v, a):
    p = 1
    q = 1
    z_factor = dot_sigmoid(xi, a)
    for t in range(len(yi)):
        e = yit_estimate(yi[t], xi, v[t, :])
        f = 1 - e
        p *= e
        q *= f
    return p * z_factor / (p * z_factor + q * (1 - z_factor))

def calc_p_tilde_tree(xi, yi, v, a):
    p = 1
    q = 1
    # something like this: 
    # z_factor = a.predict.proba(xi)
    for t in range(len(yi)):
        e = yit_estimate(yi[t], xi, v[t, :])
        f = 1 - e
        p *= e
        q *= f
    return p * z_factor / (p * z_factor + q * (1 - z_factor))


def log_loss(w, y, x, sigma):
    s = 0
    for i in range(len(y)):
        sigma = dot_sigmoid(x[i, :], w)
        s += y[i] * np.log(sigma) + (1 - y[i]) * np.log(1 - sigma)
    return -s + sigma * np.dot(w, w)


def gradient_log_loss(w, y, x, sigma):
    # x_1 = np.hstack((x, np.ones((x.shape[0], 1))))
    return (
        -sum(x[i, :] * (y[i] - dot_sigmoid(x[i, :], w)) for i in range(len(y)))
        + sigma * w
    )


def soft_lab(yit, p_tilde):
    return yit * p_tilde + (1 - yit) * (1 - p_tilde)


def real_likelihood(a, v, x, y, N, T):
    s = 0
    for i in range(N):
        xi = x[i, :]
        yi = y[i, :]
        for t in range(T):
            yit = yi[t]
            zf = dot_sigmoid(xi, a)
            yf = yit_estimate(yit, xi, v[t, :])
            s += np.log(zf * yf + (1 - zf) * (1 - yf))
    return s


def EM(x, y, epsilon_tot, sigma):
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
    while abs(l_curr - l_prev) > epsilon_tot:
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
        # a_new = scipy.optimize.minimize(
        #     log_loss,
        #     np.random.randn(D + 1),
        #     jac=gradient_log_loss,
        #     args=(p_tilde, x_1, sigma),
        #     method="L-BFGS-B",
        # ).x
        a_new = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=2020)
        a_new.fit(x_1, p_tilde)

        for t in range(T):
            v[t, :] = scipy.optimize.minimize(
                log_loss,
                v[t, :],
                jac=gradient_log_loss,
                args=(soft_label[:, t], x_1, sigma),
                method="L-BFGS-B",
            ).x

        l_curr = real_likelihood(a_new, v, x_1, y, N, T)
        ls.append(l_curr)

    return a, v, ls
