import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import tree
import data_gen
from utils import dot_sigmoid, eval_classifier
from yan_yan_et_al import yan_yan_et_al
from hyper_opt import optimize_reg_yan_yan, optimize_tree
from majority import true_classifier, concat, majority
from yan_yan_et_al_tree import yan_yan_tree

"""
from yan_yan_et_al import yan_yan_et_al
from raykar_logistic import raykar_et_al
from majority import majority, true_classifier, concat
from utils import eval_classifier, dot_sigmoid
from plotting import create_color_annotator
"""

dims = 12
n_samples = 1000
label_noise = 0.1
n_experts = 5
expert_hplane_dist = 0.5
mis_clas_sense = 1
epsilon_yan_yan = 1e-2
n_folds = 5
n_roc_samples = 10000

np.random.seed(0)

# First generate a 12-dimensional hypersphere -- our "data" cloud
hypersphere_fn = f"hypersphere{dims}_{n_samples}.csv"
if not os.path.exists(hypersphere_fn):
    data_gen.generate_hypersphere(dims, n_samples, hypersphere_fn)

hsphere = np.array(pd.read_csv(hypersphere_fn))

# Next, generate the true classifier as a hyperplane,
# classify each point and flip the label with a small probability
a_real = np.ones(dims + 1)
a_real[-1] = 0
nonoise_y = data_gen.classify_points(a_real, hsphere)
y = data_gen.noise_labels(nonoise_y, label_noise)
print(y)

# Next, generate the classfiers:
# We let them correspond to hyperplanes 1 / 2 from the origin at a random angle
# In the region outside the hyperplane they are likely to make mistakes and inside they are not
v_real = np.zeros((n_experts, dims + 1))
for i in range(n_experts):
    v_real[i, :] = data_gen.random_hyperplane(dims, expert_hplane_dist, mis_clas_sense)

# Now let the experts missclassify the points in the region outside the hyperplane with some probability
advice = np.zeros((n_samples, n_experts))
for i in range(n_experts):
    advice[:, i] = data_gen.expert_classification(v_real[i, :], hsphere, y)

inds = list(range(n_samples))
rand_inds = np.random.shuffle(inds)
train_index = inds[: 4 * n_samples // 5]
test_index = inds[4 * n_samples // 5 :]

classification_thresholds = np.linspace(-0.01, 1.00, n_roc_samples)
roc_EM_algorithm = np.zeros((n_folds, n_roc_samples, 2))
roc_EM_algorithm_reg = np.zeros((n_folds, n_roc_samples, 2))
roc_EM_algorithm_tree = np.zeros((n_folds, n_roc_samples, 2))
roc_majority = np.zeros((n_folds, n_roc_samples, 2))
roc_true = np.zeros((n_folds, n_roc_samples, 2))
roc_concat = np.zeros((n_folds, n_roc_samples, 2))

# Assemble the data
i = 0
X_train = hsphere[train_index]
advice_train = advice[train_index]
y_train = y[train_index]

ls = np.linspace(0.1, 1, 10)
a_tree, v_tree, _ = yan_yan_tree(
    X_train, advice_train, epsilon_yan_yan, 6, 0
)  # optimize_tree(X_train, advice_train, epsilon_yan_yan)

a_reg, v_reg = optimize_reg_yan_yan(X_train, advice_train, ls, epsilon_yan_yan)
a, v, l = yan_yan_et_al(X_train, advice_train, epsilon_yan_yan, 0)
w_concat = concat(X_train, advice_train, 0)
w_true = true_classifier(X_train, y_train, 0)
w_majority = majority(X_train, advice_train, 0)

y_test = y[test_index]
X_test = hsphere[test_index]
X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
advice_test = advice[test_index]
yan_yan_votes_logistically = dot_sigmoid(X_test, a)
# yan_yan_votes_logistically_reg = dot_sigmoid(X_test, a_reg)
yan_yan_votes_arboraly = a_tree.predict_proba(X_test)[:, 1]
print(yan_yan_votes_arboraly)
majority_votes = dot_sigmoid(X_test, w_majority)
true_votes = dot_sigmoid(X_test, w_true)
concat_votes = dot_sigmoid(X_test, w_concat)

for j, t in enumerate(classification_thresholds):
    TPR, FPR = eval_classifier(y_test, yan_yan_votes_logistically, t)
    roc_EM_algorithm[i, j, :] += (FPR, TPR)
    TPR, FPR = eval_classifier(y_test, yan_yan_votes_logistically_reg, t)
    roc_EM_algorithm_reg[i, j, :] += (FPR, TPR)
    TPR, FPR = eval_classifier(y_test, yan_yan_votes_arboraly, t)
    roc_EM_algorithm_tree[i, j, :] += (FPR, TPR)
    TPR, FPR = eval_classifier(y_test, majority_votes, t)
    roc_majority[i, j, :] += (FPR, TPR)
    TPR, FPR = eval_classifier(y_test, true_votes, t)
    roc_true[i, j, :] += (FPR, TPR)
    TPR, FPR = eval_classifier(y_test, concat_votes, t)
    roc_concat[i, j, :] += (FPR, TPR)
"""
p_1 = np.hstack((hsphere, np.ones((n_samples, 1))))

plt.figure(f"Fold {i}")
for i in range(n_experts):
    plt.subplot(2, 3, i + 1)
    plt.scatter(
        hsphere[train_index, 0],
        hsphere[train_index, 1],
        c=advice_train[:, i],
        cmap="coolwarm_r",
    )
plt.subplot(2, 3, 6)
plt.scatter(
    hsphere[train_index, 0],
    hsphere[train_index, 1],
    c=y_train,
    cmap="coolwarm_r",
)

plt.figure()
for i in range(n_experts):
    plt.subplot(2, 3, i + 1)
    plt.scatter(
        hsphere[train_index, 0],
        hsphere[train_index, 1],
        c=dot_sigmoid(p_1[train_index, :], v[i, :]),
        cmap="coolwarm_r",
    )
plt.subplot(2, 3, 6)
plt.scatter(
    hsphere[train_index, 0],
    hsphere[train_index, 1],
    c=dot_sigmoid(p_1[train_index, :], a),
    cmap="coolwarm_r",
)
plt.show()
"""


i = 0
plt.plot(roc_EM_algorithm[i, :, 0], roc_EM_algorithm[i, :, 1], label="EM")
plt.plot(
    roc_EM_algorithm_reg[i, :, 0], roc_EM_algorithm_reg[i, :, 1], label="EM l = 0.1"
)
plt.plot(
    roc_EM_algorithm_tree[i, :, 0], roc_EM_algorithm_tree[i, :, 1], label="EM tree"
)
plt.plot(roc_majority[i, :, 0], roc_majority[i, :, 1], label="Majority")
plt.plot(roc_concat[i, :, 0], roc_concat[i, :, 1], label="Concat")
plt.legend()

"""
plt.subplot(2, 1, 2)
roc_EM_algorithm = np.mean(roc_EM_algorithm, axis=0)
roc_EM_algorithm_reg = np.mean(roc_EM_algorithm_reg, axis=0)
roc_majority = np.mean(roc_majority, axis=0)
roc_true = np.mean(roc_true, axis=0)
roc_concat = np.mean(roc_concat, axis=0)
plt.plot(roc_EM_algorithm[:, 0], roc_EM_algorithm[:, 1], label="EM")
plt.plot(roc_EM_algorithm_reg[:, 0], roc_EM_algorithm_reg[:, 1], label="EM l = 0.1")
plt.plot(roc_majority[:, 0], roc_majority[:, 1], label="Majority")
# plt.plot(roc_true[:, 0], roc_true[:, 1], label="True")
plt.plot(roc_concat[:, 0], roc_concat[:, 1], label="Concat")
plt.legend()
p"""


plt.show()
