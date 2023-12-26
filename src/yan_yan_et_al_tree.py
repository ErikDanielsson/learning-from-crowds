import numpy as np
from utils import dot_sigmoid
from sklearn import tree
from logistic_regression import log_reg

duplications = 100


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
    z_factor = a.predict_proba([xi])[0, 1]
    for t in range(len(yi)):
        e = yit_estimate(yi[t], xi, v[t, :])
        f = 1 - e
        p *= e
        q *= f
    return p * z_factor / (p * z_factor + q * (1 - z_factor))


def s_log_loss(sigma, y):
    return y * np.log(sigma) + (1 - y) * np.log(1 - sigma)


vec_log_loss = np.vectorize(s_log_loss)


def log_loss(w, y, x):
    sigma = dot_sigmoid(x, w)
    return -sum(vec_log_loss(sigma, y))


def gradient_log_loss(w, y, x):
    s = dot_sigmoid(x, w)
    return -np.dot(x.T, y - s)


def hessian_log_loss(w, y, x):
    s = dot_sigmoid(x, w)
    S = np.diag(np.multiply(s, 1 - s))
    return np.linalg.multi_dot((x.T, S, x))


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


def real_likelihood_tree(a, v, x, y, N, T):
    s = 0
    for i in range(N):
        xi = x[i, :]
        yi = y[i, :]
        for t in range(T):
            yit = yi[t]
            zf = a.predict_proba([xi])[0, 1]
            yf = yit_estimate(yit, xi, v[t, :])
            s += np.log(zf * yf + (1 - zf) * (1 - yf))
    return s


def yan_yan_tree(x, y, epsilon_tot, depth):
    N, D = x.shape
    N, T = y.shape
    p_tilde = np.random.rand(N)
    soft_label = np.zeros((N, T))
    v = np.zeros((T, D + 1))
    v[:, -1] = 10
    l_prev = -np.inf
    l_curr = 0
    ls = []

    x_1 = np.hstack((x, np.ones((N, 1))))
    a = tree.DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=5)
    a.fit(x_1, y[:, 0])
    a_new = tree.DecisionTreeClassifier(
        criterion="entropy", random_state=10, max_depth=depth
    )
    a_new.fit(x_1, y[:, 1])

    x_dup = []
    for i in range(N):
        for _ in range(duplications):
            x_dup.append(x_1[i, :])

    while abs(l_curr - l_prev) > epsilon_tot:
        # print(f"diff: {l_curr - l_prev}")
        l_prev = l_curr
        a = a_new
        # E-step
        for i in range(N):
            xi = x_1[i, :]
            yi = y[i, :]
            p_tilde[i] = calc_p_tilde_tree(xi, yi, v, a)

        for i in range(N):
            yi = y[i, :]
            p_ti = p_tilde[i]
            for t in range(T):
                soft_label[i, t] = soft_lab(yi[t], p_ti)

        p_til_dup = np.empty(0)
        for i in range(N):
            num = int(round(p_tilde[i] * duplications))
            p_til_dup = np.concatenate(
                (p_til_dup, np.ones(num), np.zeros(duplications - num))
            )

        a_new = tree.DecisionTreeClassifier(
            criterion="entropy", random_state=10, max_depth=depth
        )
        a_new.fit(x_dup, p_til_dup)

        for t in range(T):
            v[t, :] = log_reg(soft_label[:, t], x_1, 0, D).x

        l_curr = real_likelihood_tree(a_new, v, x_1, y, N, T)
        print(l_curr)
        ls.append(l_curr)

    return a, v, ls