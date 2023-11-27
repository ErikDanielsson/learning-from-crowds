import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


def generate_data(N, w):
    x1 = np.random.uniform(0, 1, N)
    x2 = np.random.uniform(0, 1, N)
    x = np.array([x1, x2]).T
    y = 1 * (np.dot(x, w) > 0)
    return x, y


def expert_advice(y, x, w):
    N = len(y)
    M = w.shape[1]
    advice = np.zeros((N, M))
    for i, yi in enumerate(y):
        if yi == 1:
            for j in range(M):
                advice[i, j] = 1 * (
                    np.random.uniform(0, 1)
                    >= sigmoid(np.dot(w[0:2, j], x[i, :]) + w[-1, j])
                )
        else:
            for j in range(M):
                advice[i, j] = 1 * (
                    np.random.uniform(0, 1)
                    >= sigmoid(np.dot(w[0:2, j], x[i, :]) + w[-1, j])
                )
    return advice


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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
        p *= yit_estimate(yit, xi, v[t, :]) * z_factor
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


def log_loss(y, x, w):
    return -sum(
        y[i] * np.log(dot_sigmoid(x[i, :], w))
        + (1 - y[i]) * np.log(1 - dot_sigmoid(x[i, :], w))
        for i in range(len(y))
    )


def gradient_log_loss(y, x, w):
    x_1 = np.hstack((x, np.ones((x.shape[0], 1))))
    return -sum(x_1[i, :] * (y[i] - dot_sigmoid(x[i, :], w)) for i in range(len(y)))


def soft_lab(yit, p_tilde):
    return yit * p_tilde + (1 - yit) * (1 - p_tilde)


def EM(x, y, epsilon_tot, epsilon_log):
    N, D = x.shape
    N, T = y.shape
    p_tilde = np.zeros(N)
    soft_label = np.zeros((N, T))
    v = np.zeros((T, D + 1))
    a = np.zeros(D + 1)
    a_new = np.empty(D + 1, dtype=float)
    a_new.fill(1)
    gamma = 0.01
    x_1 = np.hstack((x, np.ones((N, 1))))
    while abs(np.linalg.norm(a) - np.linalg.norm(a_new)) > epsilon_tot:
        a = a_new
        # E-step
        for i in range(N):
            xi = x[i, :]
            yi = y[i, :]
            p_tilde[i] = calc_p_tilde(xi, yi, v, a)
            p_ti = p_tilde[i]
            for t in range(T):
                yit = yi[t]
                soft_label[i, t] = soft_lab(yit, p_ti)

        # M-step
        a_new = scipy.optimize.minimize(
            lambda w: log_loss(p_tilde, x, w),
            a_new,
            jac=lambda w: gradient_log_loss(p_tilde, x, w),
            method="BFGS",
        ).x
        a_new /= a_new[0]
        for t in range(T):
            v[t, :] = scipy.optimize.minimize(
                lambda w: log_loss(soft_label[:, t], x, w),
                v[t, :],
                jac=lambda w: gradient_log_loss(soft_label[:, t], x, w),
                method="BFGS",
            ).x
            v[t, :] /= v[t, 0]

        print(a)
        print(v)
        """
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
                new_v = v[t, :] - gamma * np.linalg.inv(h) @ g(
                    v[t, :], soft_label[:, t], x_1, N
                )
                new_v /= np.linalg.norm(new_v)
                if np.linalg.norm(new_v - v[t, :]) < epsilon_log:
                    break
                v[t, :] = new_v
            v[t, :] /= v[t, 0]
        """
    return a, v


w_real = np.array([1, -2])
x, y = generate_data(10000, w_real)
advice = expert_advice(y, x, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 200]]))
a, v = EM(x, advice, 1e-6, 1e-6)
print(a, v)
positive = np.array([[x1, x2] for (x1, x2), yi in zip(x, y) if yi == 1])
negative = np.array([[x1, x2] for (x1, x2), yi in zip(x, y) if yi == 0])
plt.scatter(positive[:, 0], positive[:, 1])
plt.scatter(negative[:, 0], negative[:, 1])
rot90 = np.array([[0, -1], [1, 0]])
l_real = rot90 @ w_real
l_est = rot90 @ a[0:2]
plt.plot(np.linspace(0, 1, 100), l_real[1] / l_real[0] * np.linspace(0, 1, 100))
plt.plot(np.linspace(0, 1, 100), l_est[1] / l_est[0] * np.linspace(0, 1, 100))
plt.show()
