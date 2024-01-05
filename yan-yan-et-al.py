import numpy as np
import scipy.optimize
from utils import *
from multiprocessing import Pool
from sklearn import tree
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from plotting import *

np.random.seed(33)
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


# def s_log_loss(sigma, y):
#     return y * np.log(sigma) + (1 - y) * np.log(1 - sigma)


# vec_log_loss = np.vectorize(s_log_loss)


def log_loss(w, y, x):
    s = 0
    for i in range(len(y)):
        sigma = dot_sigmoid(x[i, :], w)
        s += y[i] * np.log(sigma) + (1 - y[i]) * np.log(1 - sigma)
    return -s


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


def EM(x, y, epsilon_tot, depth):
    N, D = x.shape
    N, T = y.shape
    p_tilde = np.random.rand(N)
    soft_label = np.zeros((N, T))
    v = np.zeros((T, D + 1))
    v[:, -1] = 10
    l_prev = -100000000
    l_curr = -100000
    ls = []

    x_1 = np.hstack((x, np.ones((N, 1))))
    a = tree.DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=depth)
    a.fit(x_1, y[:, 0])
    a_new = tree.DecisionTreeClassifier(
        criterion="entropy", random_state=10, max_depth=5
    )
    a_new.fit(x_1, y[:, 1])
    # a = np.ones((D+1,1))
    # a_new = np.zeros((D+1,1))

    x_dup = []
    for i in range(N):
        for _ in range(duplications):
            x_dup.append(x_1[i, :])

    while l_curr - l_prev > epsilon_tot:
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
            v[t, :] = scipy.optimize.minimize(
                log_loss,
                v[t,:],
                jac=gradient_log_loss,
                hess=hessian_log_loss,
                args=(soft_label[:, t], x_1),
                method="trust-exact",
            ).x
        print(v[0, :])

        l_curr = real_likelihood_tree(a_new, v, x_1, y, N, T)
        print(l_curr)
        ls.append(l_curr)

    return a, v, ls


a_real = np.array([0, 1, -0.2])
N = 1000
x, y = generate_data(N, a_real)
x = x[:, 0:-1]
print(x[0], y[0])
v_real = np.array(
    [
        [10, -3, 1],
        [10, 10, -10],
        [4, 4, -2],
        [0, -4, 2],
    ]
)
plt.figure()
plot_bin_datapoints(x, y, plt.gca(), marker="o", markersize=20)
plot_line(a_real, plt.gca(), color="green")
plt.gcf().suptitle("Ground truth", fontsize=16)
plt.gca().set_xlabel("Feature 1")
plt.gca().set_ylabel("Feature 2")


advice = expert_advice(y, x, v_real)

annotator_inds = [(0, 0), (1, 0), (0, 1), (1, 1)]

# fig, axs = plt.subplots(2, 2)
# for i, ind in enumerate(annotator_inds):
#     plot_bin_datapoints(x, advice[:, i], axs[ind], marker="o")
#     plot_line(v_real[i, :], axs[ind], color="green")
#     axs[ind].set_title(f"Expert {i + 1}")
#     axs[ind].set_xlabel("Feature 1")
#     axs[ind].set_ylabel("Feature 2")
# fig.suptitle("Expert predictions", fontsize=16)
# fig.tight_layout()
plt.show()

# n_folds = 5
# max_depth = np.array([(i+1) for i in range(int(np.log2(x.shape[0])))])
# print(len(max_depth))
# kf = KFold(n_splits=n_folds, shuffle=False)  # , random_state=random_seed)
# folds = list(kf.split(x))
# #alphas = np.exp(-10 * np.linspace(0, 1, 5))
# likelihoods = np.zeros(len(max_depth))
# for i, (train_index, test_index) in enumerate(folds):
#     X_train = x[train_index, :]
#     advice_train = advice[train_index, :]
#     X_test = x[test_index, :]
#     advice_test = advice[test_index, :]
#     N, T = advice_test.shape
#     X_test = np.hstack((X_test, np.ones((N, 1))))
#         # for i, alpha in enumerate(alphas):
#         #     a, v, _ = yan_yan_tree(X_train, advice_train, epsilon, max_depth, alpha)
#         #     print(
#         #         a.get_depth(),
#         #         alpha,
#         #         real_likelihood_tree(a, v, X_test, advice_test, N, T),
#         #     )
#         #     likelihoods[i] += real_likelihood_tree(a, v, X_test, advice_test, N, T)
#     for j in max_depth:
#         a, v, _ = EM(X_train, advice_train, 1, j)
#         likelihoods[j-1] += real_likelihood_tree(a, v, X_test, advice_test, N, T)
#     print('done')
# best_ind = np.argmax(likelihoods)
# print(likelihoods)
# print(max_depth[best_ind])
a, v, _ = EM(x, advice, 1e-2, 1)
#a, v, ls = EM(x, advice, 1, 1e-6)
print(v)
tree.plot_tree(a, class_names=True)
plt.show()

N = 1000
plt.figure()
x_1 = np.hstack((x, np.ones((N, 1))))
pred = a.predict(x_1)
plot_bin_datapoints(x, pred, plt.gca(), markersize=20)

plt.gcf().suptitle("Estimated ground truth", fontsize=16)
plt.gca().set_xlabel("Feature 1")
plt.gca().set_ylabel("Feature 2")

# fig, axs = plt.subplots(2, 2)
# for i, ind in enumerate(annotator_inds):
#     conf = create_color_classifier(x_1, v[i, :])
#     plot_cont_datapoints(x, conf, axs[ind])
#     axs[ind].set_title(f"Expert {i + 1}")
#     axs[ind].set_xlabel("Feature 1")
#     axs[ind].set_ylabel("Feature 2")
# fig.suptitle("Estimated expert bias", fontsize=16)
# fig.tight_layout()
# fig, axs = plt.subplots(2, 2)
# for i, ind in enumerate(annotator_inds):
#     conf = create_color_classifier(x_1, v_real[i, :])
#     plot_cont_datapoints(x, conf, axs[ind])
#     axs[ind].set_title(f"Expert {i + 1}")
#     axs[ind].set_xlabel("Feature 1")
#     axs[ind].set_ylabel("Feature 2")
# fig.suptitle("Expert bias", fontsize=16)
# fig.tight_layout()
plt.show()
