import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

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


def EM(X, y, epsilon_tot, epsilon_log):
    N, n_features = X.shape
    R = y.shape[1]
    w = np.zeros(n_features)
    # Majority voting for initalization
    mu = np.zeros(N)
    for i in range(N):
        mu[i] = 1 / R * sum(y[i, :])

    alpha = np.zeros(R)
    beta = np.zeros(R)
    for j in range(R):
        alpha[j] = sum(mu[i] * y[i, j] for i in range(N)) / sum(mu)
        beta[j] = sum((1 - mu[i]) * (1 - y[i, j]) for i in range(N)) / (N - sum(mu))
    eta = 0.01
    print(mu)
    for _ in range(100):
        # E-step
        p = np.zeros(N)
        a = np.zeros(N)
        b = np.zeros(N)
        for i in range(N):
            p[i] = sigmoid(np.dot(w, X[i, :]))
            a[i] = np.prod(
                [alpha[j] if y[i, j] == 1 else 1 - alpha[j] for j in range(R)]
            )
            b[i] = np.prod([beta[j] if y[i, j] == 0 else 1 - beta[j] for j in range(R)])
            mu[i] = a[i] * p[i] / (a[i] * p[i] + b[i] * (1 - p[i]))

        # M-step
        # For the analytical parameters
        for j in range(R):
            alpha[j] = sum(mu[i] * y[i, j] for i in range(N)) / sum(mu)
            beta[j] = sum((1 - mu[i]) * (1 - y[i, j]) for i in range(N)) / (N - sum(mu))
        # Newton-Raphson for the logistic regression
        for _ in range(100):
            h = H(w, X, N)
            new_w = w - eta * np.linalg.inv(h) @ g(w, mu, X, N)
            new_w /= np.linalg.norm(w)
            if np.linalg.norm(new_w - w) < epsilon_log:
                w = new_w
                break

    return alpha, beta, w


w = np.array([1, -2])
x, y = generate_data(1000, w)
advice = expert_advice(y, [1, 0.5, 1], [1, 1, 1])
positive = np.array([[x1, x2] for (x1, x2), yi in zip(x, y) if yi == 1])
negative = np.array([[x1, x2] for (x1, x2), yi in zip(x, y) if yi == 0])
plt.scatter(positive[:, 0], positive[:, 1])
plt.scatter(negative[:, 0], negative[:, 1])
rot90 = np.array([[0, -1], [1, 0]])
l = rot90 @ w
plt.plot(np.linspace(0, 1, 100), l[1] / l[0] * np.linspace(0, 1, 100))
# plt.show()
alpha, beta, w = EM(x, advice)
print(alpha, beta, w)
