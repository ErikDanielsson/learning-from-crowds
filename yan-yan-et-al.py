import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# We define v = [w \gamma]
def dot_sigmoid(x, v):
    return sigmoid(np.dot(x, v[:-1]) + v[-1])


def yit_estimate(yit, xi, v):
    return 1 - dot_sigmoid(xi, v) if yit == 0 else dot_sigmoid(xi, v)


def yit_est_gauss(yit,xi,v):
    sigma = dot_sigmoid(xi,v) 
    const = 1 / ( sigma * np.sqrt(2*np.pi) )
    return const * np.exp(1 / sigma^2) if y == 1 else const * np.exp(-1/( 2 * sigma^2 ))


def calc_p_tilde(xi, yi, v, a):
    p = 1
    z_factor = dot_sigmoid(xi, a)
    for t in range(len(yi)):
        yit = yi[t]
        p *= yit_estimate(yit, xi, v) * z_factor
    return p
# should we change this ^ to be flexible to bernoulli or gaussian?

def partial_f_partial_a(xi, pi, a):
    a_len = len(a)
    s = np.zeros(a_len)
    for i in range(xi.shape[0]):
        xi_n_one = np.ones(a_len)
        xi_n_one[:-1] = xi[i, :]
        delta_p = 2 * pi - 1
        s_eval = dot_sigmoid(xi, a)
        s += delta_p * s_eval * (s_eval - 1) * xi_n_one
    return s


def partial_f_partial_eta_t(yit, pi):
    delta = 1 - 2 * pi
    return -delta if yit == 1 else delta


def partial_eta_t_v(xi, v):
    xi_n_one = np.ones(len(v))
    xi_n_one[:-1] = xi
    s_eval = dot_sigmoid(xi, v)
    return s_eval * (1 - s_eval) * xi_n_one


def partial_f_partial_sigma_t(yit, pi, xi, v):
    sigma = dot_sigmoid(xi,v)
    return (1 - pi) / sigma^3 - 1/sigma if y==1 else pi / sigma^3 - 1/sigma


def grad_f_opt(x, y, p_tilde, v, a):
    N, D = x.shape
    N, T = y.shape
    gradient = np.zeros(D + 1 + T * (D + 1))
    for i in range(N):
        xi = x[i, :]
        pi = p_tilde[i]
        gradient[: D + 1] += partial_f_partial_a(xi, pi)
        for t in range(T):
            yit = y[i, t]
            pfpe = partial_f_partial_eta_t(yit, pi)
            pepv = partial_eta_t_v(xi, v)
            gradient[(t + 1) * (D + 1) : (t + 2) * (D + 1)] += pfpe * pepv
    return gradient


def f(x, y, p_tilde, theta):
    N, D = x.shape
    N, T = y.shape
    v = np.zeros(T, D + 1)
    a = np.zeros(D + 1)
    a[:] = theta[: D + 1]
    for t in range(T):
        v[t, :] = theta[t * (D + 1) : (t + 1) * (D + 1)]
    return f_opt(x, y, p_tilde, v, a)


def f_opt(x, y, p_tilde, v, a):
    N, D = x.shape
    N, T = y.shape
    l = 0
    for i in range(N):
        xi = x[i, :]
        yi = y[i, :]
        classifier_eval = dot_sigmoid(x, a)
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

        minimize(lambda theta: f(x, y, p_tilde, theta))
