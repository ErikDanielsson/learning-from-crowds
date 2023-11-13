function generate_data(N, w)
    x1 = np.random.uniform(0, 1, N)
    x2 = np.random.uniform(0, 1, N)
    x = [x1, x2]
    y = 1 * (np.dot(x, w) > 0)
    return x, y
end


function expert_advice(y, alpha, beta)
    N = len(y)
    M = len(alpha)
    advice = np.zeros((N, M))
    for i, yi in enumerate(y):
        if yi == 1:
            for j, a in enumerate(alpha):
                advice[i, j] = 1 * (np.random.uniform(0, 1) <= a)
            end
        else:
            for j, b in enumerate(beta):
                advice[i, j] = 1 * (np.random.uniform(0, 1) >= b)
            end
        end
    end
    return advice
end


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


def EM(X, advice):
    N, n_features = X.shape
    n_experts = advice.shape[1]
    w = np.zeros(n_features)
    # Majority voting for initalization
    alpha = np.random.rand(n_experts)
    beta = np.random.rand(n_experts)
    eta = np.ones(n_features)
    for _ in range(100):
        # E-step
        p = np.zeros(N)
        a = np.zeros(N)
        b = np.zeros(N)
        mu = np.zeros(N)
        for i in range(N):
            p[i] = sigmoid(np.dot(w, X[i, :]))
            a[i] = np.prod(
                [
                    alpha[j] if advice[i, j] == 1 else 1 - alpha[j]
                    for j in range(n_experts)
                ]
            )
            b[i] = np.prod(
                [
                    beta[j] if advice[i, j] == 0 else 1 - beta[j]
                    for j in range(n_experts)
                ]
            )
            mu[i] = a[i] * p[i] / (a[i] * p[i] + b[i] * (1 - p[i]))

        # M-step
        # For the analytical parameters
        for j in range(n_experts):
            alpha[j] = sum(mu[i] * advice[i, j] for i in range(N)) / sum(mu)
            beta[j] = sum((1 - mu[i]) * (1 - advice[i, j]) for i in range(N)) / (
                N - sum(mu)
            )
        # Newton-Raphson for the logistic regression
        for i in range(10):
            w = w - eta @ np.linalg.inv(H(w, X, N)) @ g(w, mu, X, N)

    return alpha, beta, w


w = np.array([1, -1])
x, y = generate_data(100, w)
advice = expert_advice(y, [1, 1, 1], [1, 1, 1])
positive = np.array([[x1, x2] for (x1, x2), yi in zip(x, y) if yi == 1])
negative = np.array([[x1, x2] for (x1, x2), yi in zip(x, y) if yi == 0])
plt.scatter(positive[:, 0], positive[:, 1])
plt.scatter(negative[:, 0], negative[:, 1])
rot90 = np.array([[0, -1], [1, 0]])
l = rot90 @ w
plt.plot(np.linspace(0, 1, 100), l[1] / l[0] * np.linspace(0, 1, 100))
plt.show()
alpha, beta, w = EM(x, advice)
print(alpha, beta, w)
