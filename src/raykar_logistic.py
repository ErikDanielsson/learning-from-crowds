import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import log_reg

random.seed(123)


def generate_data(N, w):
    x1 = np.random.uniform(0, 1, N)
    x2 = np.random.uniform(0, 1, N)
    x = np.array([x1, x2]).T
    y = 1 * (np.dot(x, w) > 0)
    return x, y


def expert_advice(y, alpha, beta):
    N = len(y)
    M = len(alpha)
    advice = np.zeros((N, M))
    for i, yi in enumerate(y):
        if yi == 1:
            for j, a in enumerate(alpha):
                advice[i, j] = 1 * (np.random.uniform(0, 1) <= a)
        else:
            for j, b in enumerate(beta):
                advice[i, j] = 1 * (np.random.uniform(0, 1) >= b)
    return advice


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def g(w, mu, X, N):
    return sum((mu[i] - sigmoid(np.dot(w, X[i, :]))) * X[i, :] for i in range(N))


def H(w, X, N):
    return -sum(
        sigmoid(np.dot(w, X[i, :]))
        * (1 - sigmoid(np.dot(w, X[i, :])))
        * np.outer(X[i, :], X[i, :])
        for i in range(N)
    )


def raykar_et_al(X, y, epsilon_tot, epsilon_log):
    N, n_features = X.shape
    R = y.shape[1]
    w = np.zeros(n_features + 1)
    # Majority voting for initalization
    mu = np.zeros(N)
    l_obs_prev = -10000000
    for i in range(N):
        mu[i] = 1 / R * sum(y[i, :])

    alpha = np.zeros(R)
    beta = np.zeros(R)
    for j in range(R):
        alpha[j] = sum(mu[i] * y[i, j] for i in range(N)) / sum(mu)
        beta[j] = sum((1 - mu[i]) * (1 - y[i, j]) for i in range(N)) / (N - sum(mu))
    X_1 = np.hstack((X, np.ones((N, 1))))
    eta = 0.01
    for k in range(10000):
        if k % 10 == 0:
            print(f"Iteration {k}")
        # E-step
        p = np.zeros(N)
        a = np.zeros(N)
        b = np.zeros(N)
        for i in range(N):
            p[i] = sigmoid(np.dot(w, X_1[i, :]))
            a[i] = np.prod(
                [alpha[j] if y[i, j] == 1 else 1 - alpha[j] for j in range(R)]
            )
            b[i] = np.prod([beta[j] if y[i, j] == 0 else 1 - beta[j] for j in range(R)])
            mu[i] = a[i] * p[i] / (a[i] * p[i] + b[i] * (1 - p[i]))
        l_obs = sum(np.log((a[i] * p[i]) + b[i] * (1 - p[i])) for i in range(N))
        if l_obs - l_obs_prev < epsilon_tot:
            break
        l_obs_prev = l_obs

        # M-step
        # For the analytical parameters
        for j in range(R):
            alpha[j] = sum(mu[i] * y[i, j] for i in range(N)) / sum(mu)
            beta[j] = sum((1 - mu[i]) * (1 - y[i, j]) for i in range(N)) / (N - sum(mu))
        # Newton-Raphson for the logistic regression
        w = log_reg(mu, X_1, 0, n_features).x
    return w, alpha, beta

