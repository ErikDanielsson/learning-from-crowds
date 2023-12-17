import numpy as np
import scipy.optimize
from utils import *
from multiprocessing import Pool
from sklearn import tree

np.random.seed(102)

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
    z_factor = a.predict_proba([xi])[0,1]
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
            zf = a.predict_proba([xi])[0,1]
            yf = yit_estimate(yit, xi, v[t, :])
            s += np.log(zf * yf + (1 - zf) * (1 - yf))
    return s


def EM(x, y, epsilon_tot, epsilon_log):
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
    a = tree.DecisionTreeClassifier(criterion='entropy', random_state=10, max_depth=5)
    a.fit(x_1,y[:,0])
    a_new = tree.DecisionTreeClassifier(criterion='entropy', random_state=10, max_depth=5)
    a_new.fit(x_1,y[:,1])
    # a = np.ones((D+1,1))
    # a_new = np.zeros((D+1,1))

    x_dup = []
    for i in range(len(x_1)):
        for j in range(100):
            x_dup.append(x_1[i])

    while abs(l_curr - l_prev) > epsilon_tot:
        # print(f"diff: {l_curr - l_prev}")
        l_prev = l_curr
        a = a_new
        # E-step
        for i in range(N):
            xi = x_1[i, :]
            yi = y[i, :]
            p_tilde[i] = calc_p_tilde_tree(xi, yi, v, a)
        print(p_tilde[0])

        for i in range(N):
            yi = y[i, :]
            p_ti = p_tilde[i]
            for t in range(T):
                soft_label[i, t] = soft_lab(yi[t], p_ti)

        
        p_til_dup = []
        for i in range(len(p_tilde)):
            num = int(round(p_tilde[i] * 100))
            for j in range(num):
                p_til_dup.append([1])
            for j in range(100-num):
                p_til_dup.append([0])
        # M-step
        # a_new = scipy.optimize.minimize(
        #     log_loss,
        #     np.random.randn(D + 1),
        #     jac=gradient_log_loss,
        #     args=(p_tilde, x_1),
        #     method="L-BFGS-B",
        # ).x
        # idea:
        a_new = tree.DecisionTreeClassifier(criterion='entropy', random_state=10, max_depth=5)
        a_new.fit(x_dup, p_til_dup)

        for t in range(T):
            v[t, :] = scipy.optimize.minimize(
                log_loss,
                v[t, :],
                jac=gradient_log_loss,
                args=(soft_label[:, t], x_1),
                method="L-BFGS-B",
            ).x
        print(v[0,:])

        l_curr = real_likelihood_tree(a_new, v, x_1, y, N, T)
        print(l_curr)
        ls.append(l_curr)

    return a, v, ls


a_real = np.array([0, 1, -0.5])
N = 1000
x, y = generate_data(N, a_real)
x = x[:, 0:-1]
print(x[0],y[0])
v_real = np.array(
    [
        [10, -3, 1],
        [10, 10, -10],
        [4, 4, -2],
        [0, -4, 2],
    ]
)
advice = expert_advice(y, x, v_real)
a, v, ls = EM(x, advice, 1, 1e-6)
print(v)
tree.plot_tree(a)