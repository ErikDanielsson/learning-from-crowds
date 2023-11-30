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
    x = np.hstack((x, np.ones((N, 1))))
    advice = np.zeros((N, M))
    for i, yi in enumerate(y):
        if yi == 1:
            for j in range(M):
                advice[i, j] = 1 * (
                    np.random.uniform(0, 1) <= dot_sigmoid(x[i, :], w[:, j])
                )
        else:
            for j in range(M):
                advice[i, j] = 1 * (
                    np.random.uniform(0, 1) >= dot_sigmoid(x[i, :], w[:, j])
                )
    return advice


def sigmoid(x):
    if x < -500:
        return 0
    return 1 / (1 + np.exp(-x))


# We define v = [w \gamma]
def dot_sigmoid(x, v):
    return sigmoid(np.dot(x, v))


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


def log_loss(w, y, x):
    s = 0
    for i in range(len(y)):
        sigma = dot_sigmoid(x[i, :], w)
        s += y[i] * np.log(sigma) + (1 - y[i]) * np.log(1 - sigma)
    return -s


def gradient_log_loss(w, y, x):
    # x_1 = np.hstack((x, np.ones((x.shape[0], 1))))
    return -sum(x[i, :] * (y[i] - dot_sigmoid(x[i, :], w)) for i in range(len(y)))


def soft_lab(yit, p_tilde):
    return yit * p_tilde + (1 - yit) * (1 - p_tilde)


def EM(x, y, epsilon_tot, epsilon_log):
    N, D = x.shape
    N, T = y.shape
    p_tilde = np.random.rand(N)
    soft_label = np.zeros((N, T))
    v = np.zeros((T, D + 1))
    v[:, -1] = 10
    a = np.zeros(D + 1)
    a_new = np.ones(D + 1)

    gamma = 0.01
    x_1 = np.hstack((x, np.ones((N, 1))))
    while np.linalg.norm(a - a_new) > epsilon_tot:
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
        a_new = scipy.optimize.minimize(
            log_loss,
            a_new,
            jac=gradient_log_loss,
            args=(p_tilde, x_1),
            method="L-BFGS-B",
        ).x
        a_new /= a_new[0]

        for t in range(T):
            v[t, :] = scipy.optimize.minimize(
                log_loss,
                v[t, :],
                jac=gradient_log_loss,
                args=(soft_label[:, t], x_1),
                method="L-BFGS-B",
            ).x

        print(a_new)
        print(v)

    return a, v


w_real = np.array([1, -2])
x, y = generate_data(10000, w_real)
advice = expert_advice(y, x, np.array([[0, 0, 10], [10, -10, 0], [0, 0, 10]]).T)
fig, axs = plt.subplots(2, 2)

positive = np.array([[x1, x2] for (x1, x2), yi in zip(x, y) if yi == 1])
negative = np.array([[x1, x2] for (x1, x2), yi in zip(x, y) if yi == 0])
axs[0, 0].scatter(positive[:, 0], positive[:, 1])
axs[0, 0].scatter(negative[:, 0], negative[:, 1])
positive = []
negative = []
for t in range(3):
    positive.append(
        np.array([[x1, x2] for (x1, x2), yi in zip(x, advice[:, t]) if yi == 1])
    )
    negative.append(
        np.array([[x1, x2] for (x1, x2), yi in zip(x, advice[:, t]) if yi == 0])
    )
axs[1, 0].scatter(positive[0][:, 0], positive[0][:, 1])
axs[1, 0].scatter(negative[0][:, 0], negative[0][:, 1])
axs[0, 1].scatter(positive[1][:, 0], positive[1][:, 1])
axs[0, 1].scatter(negative[1][:, 0], negative[1][:, 1])
axs[1, 1].scatter(positive[2][:, 0], positive[2][:, 1])
axs[1, 1].scatter(negative[2][:, 0], negative[2][:, 1])
plt.show()

a, v = EM(x, advice, 1e-3, 1e-6)
print(a, v)

# rot90 = np.array([[0, -1], [1, 0]])
# l_real = rot90 @ w_real
# l_est = rot90 @ a[0:2]
# axs[0, 0].plot(np.linspace(0, 1, 100), l_real[1] / l_real[0] * np.linspace(0, 1, 100))
# axs[0, 0].plot(np.linspace(0, 1, 100), l_est[1] / l_est[0] * np.linspace(0, 1, 100))


N = x.shape[0]
fig, axs = plt.subplots(2, 2)
b = np.ones(N)
x_int = np.column_stack((x,b))
pos_pred = np.array([[x1, x2] for (x1, x2), i in zip(x,range(N)) if dot_sigmoid(x_int[i,:],a) >= 0.5])
neg_pred = np.array([[x1, x2] for (x1, x2), i in zip(x,range(N)) if dot_sigmoid(x_int[i,:],a) < 0.5])
axs[0, 0].scatter(pos_pred[:, 0], pos_pred[:, 1])
axs[0, 0].scatter(neg_pred[:, 0], neg_pred[:, 1])
pos_pred = []
neg_pred = []
pred_advice = expert_advice(y,x,v.T)

for t in range(3):
    pos_pred.append(
        np.array([[x1, x2] 
                  for (x1, x2), yi, i in zip(x, y, range(N)) if (dot_sigmoid(x_int[i,:],v[t,:]))**yi * (1-dot_sigmoid(x_int[i,:],v[t,:]))**(1-yi) >= 0.5])
    )
    neg_pred.append(
        np.array([[x1, x2] 
                  for (x1, x2), yi, i in zip(x, y, range(N)) if (dot_sigmoid(x_int[i,:],v[t,:]))**yi * (1-dot_sigmoid(x_int[i,:],v[t,:]))**(1-yi) < 0.5])
    )

axs[1, 0].scatter(pos_pred[0][:, 0], pos_pred[0][:, 1])
axs[1, 0].scatter(neg_pred[0][:, 0], neg_pred[0][:, 1])
axs[0, 1].scatter(pos_pred[1][:, 0], pos_pred[1][:, 1])
axs[0, 1].scatter(neg_pred[1][:, 0], neg_pred[1][:, 1])
axs[1, 1].scatter(pos_pred[2][:, 0], pos_pred[2][:, 1])
axs[1, 1].scatter(neg_pred[2][:, 0], neg_pred[2][:, 1])
plt.show()
