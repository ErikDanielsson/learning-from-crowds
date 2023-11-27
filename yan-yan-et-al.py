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


def g(w, mu, X, N):
    return sum((mu[i] - sigmoid(np.dot(w, X[i, :]))) * X[i, :] for i in range(N))


def H(w, X, N):
    return -sum(
        sigmoid(np.dot(w, X[i, :]))
        * (1 - sigmoid(np.dot(w, X[i, :])))
        * np.outer(X[i, :], X[i, :])
        for i in range(N)
    )


def soft_lab(yit, p_tilde):
    return yit * p_tilde + (1-yit) * (1-p_tilde)


def EM(x, y, epsilon_tot, epsilon_log):
    N, D = x.shape
    N, T = y.shape
    p_tilde = np.zeros(N)
    soft_label = np.zeros(N,T)
    v = np.zeros(T, D + 1)
    a = np.zeros(D + 1)
    a_new = np.empty(D + 1, dtype = float)
    a_new.fill(1000)
    # gamma is step size
    gamma = 0.01
    x_1 = np.hstack((x,np.ones(N,1)))
    while ((np.linalg.norm(a)- np.linalg.norm(a_new))^2 > epsilon_tot):
        a = a_new
        # E-step
        for i in range(N):
            xi = x[i, :]
            yi = y[i, :]
            p_tilde[i] = calc_p_tilde(xi, yi, v, a)
            p_ti = p_tilde[i]
            for t in range(T):
                yit = yi[t]
                soft_label[i,t] = soft_lab(yit, p_ti)
        
        # M-step
        for _ in range(1000):
            h = H(a, x_1, N)
            new_a = a - gamma * np.linalg.inv(h) @ g(a, p_tilde, x_1, N)
            new_a /= np.linalg.norm(new_a)
            if np.linalg.norm(new_a - a) < epsilon_log:
                break
            a_new = new_a
        a_new /= a_new[0]
        for t in range(T):
            for _ in range(1000):
                h = H(v[t, :], x_1, N)
                new_v = v[t, :] - gamma * np.linalg.inv(h) @ g(v[t, :], soft_label[:, t], x_1, N)
                new_v /= np.linalg.norm(new_v)
                if np.linalg.norm(new_v - v[t, :]) < epsilon_log:
                    break
                v[t, : ] = new_v
            v[t, :] /= v[t, 0]
    return a, v
