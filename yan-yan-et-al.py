import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# We define v = [w \gamma]
def dot_sigmoid(x, v):
    return sigmoid(np.dot(x, v[:-1]) + v[-1])


def yit_estimate(yit, xi, v):
    return 1 - dot_sigmoid(xi, v) if yit == 0 else dot_sigmoid(xi, v)


def calc_p_tilde(xi, yi, v, a):
    p = 1
    z_factor = dot_sigmoid(xi, a)
    for t in range(len(yi)):
        yit = yi[t]
        p *= yit_estimate(yit, xi, v) * z_factor
    return p


def partial_f_partial_a(x, yi, p_tilde, vt, a):
    a_len = len(a)
    s = np.zeros(a_len)
    for i in range(x.shape[0]):
        xi = np.ones(a_len)
        xi[:-1] = x[i, :]
        p = p_tilde(xi, yi, vt, a)
        delta_p = 2 * p - 1
        s += delta_p * dot_sigmoid(x, vt) * xi
    return s


def partial_f_partial_eta_t(xi, yit, vt, a):
    p = calc_p_tilde(
        xi,
        yit,
    )
    if yit == 1:
        return

def grad_f_opt(x, y, p_tilde, v, a):


def f_opt(x, y, p_tilde, v, a):
    N, D = x.shape
    N, T = y.shape
    l = 0
    for i in range(N):
        xi = x[i, :]
        yi = y[i, :]
        classifier_eval = dot_sigmoid(x, v)
        for t in range(T):
            yit = yi[t]
            yit_eval = yit_estimate(yit, xi, v)
            # Contributions corresponding to z = 1
            l += (np.log(yit_eval) + np.log(classifier_eval)) * p_tilde[i]
            # Contributions corresponding to z = 0
            l += (np.log(1 - yit_eval) + np.log(1 - classifier_eval)) * (1 - p_tilde[i])
    return l


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
