import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_gen
from itertools import groupby
from statistics import mean
from utils import dot_sigmoid, eval_classifier
from yan_yan_et_al import yan_yan_et_al
from hyper_opt import optimize_reg_yan_yan, optimize_tree
from majority import true_classifier, concat, majority
from yan_yan_et_al_tree import yan_yan_tree
from raykar_logistic import raykar_et_al

dims = 9
n_samples = 1000
label_noise = 0.1
n_experts = 5
expert_hplane_dist = 0.7
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


# Assemble the data
i = 0
X_train = hsphere[train_index]
advice_train = advice[train_index]
y_train = y[train_index]

ls = np.exp(np.linspace(-5, 2, 10))
classifiers = {}
a_tree, v_tree, max_depth = optimize_tree(X_train, advice_train, epsilon_yan_yan)
classifiers["yan-yan-tree"] = (a_tree, False)

a_reg, v_reg, lamd = optimize_reg_yan_yan(X_train, advice_train, ls, epsilon_yan_yan)
classifiers["yan-yan-reg"] = (a_reg, True)
a, v, l = yan_yan_et_al(X_train, advice_train, epsilon_yan_yan, 0)
classifiers["yan-yan"] = (a, True)
w_raykar, alpha, beta = raykar_et_al(X_train, advice_train, epsilon_yan_yan, 0)
classifiers["raykar"] = (w_raykar, True)
w_concat = concat(X_train, advice_train, 0)
classifiers["concat"] = (w_concat, True)
# w_true = true_classifier(X_train, y_train, 0)
# classifiers["true"] = (w_true, True)
w_majority = majority(X_train, advice_train, 0)
classifiers["majority"] = (w_majority, True)

y_test = y[test_index]
X_test_1 = hsphere[test_index]
X_test_1 = np.concatenate((X_test_1, np.ones((X_test_1.shape[0], 1))), axis=1)
advice_test = advice[test_index]

prob_predictions = {}
for name, (classifier, is_LR) in classifiers.items():
    if is_LR:
        prob_predictions[name] = dot_sigmoid(X_test_1, classifier)
    else:
        prob_predictions[name] = classifier.predict_proba(X_test_1)[:, 1]


classification_thresholds = np.linspace(-0.01, 1.00, n_roc_samples)
rocs = {}
for name, preds in prob_predictions.items():
    roc = np.zeros((n_roc_samples, 2))
    for j, t in enumerate(classification_thresholds):
        roc[j, :] = eval_classifier(y_test, preds, t)
    it = groupby(roc, key=lambda x: x[1])
    rocs[name] = np.array([[x, mean(yi[0] for yi in y)] for x, y in it])


plot_config = {
    "raykar": {
        "label": "Raykar et al. (LR)",
        "color": "tab:blue",
        "linestyle": "-",
    },
    "yan-yan": {
        "label": "Yan Yan et al. (LR)",
        "color": "tab:orange",
        "linestyle": "-",
    },
    "yan-yan-tree": {
        "label": f"Yan Yan et al. (tree) Max depth={max_depth}",
        "color": "tab:green",
        "linestyle": "-",
    },
    "yan-yan-reg": {
        "label": rf"Yan Yan et al. (LR reg.) $\lambda={lamd:.2f}$",
        "color": "tab:red",
        "linestyle": "-",
    },
    "concat": {
        "label": "Concat",
        "color": "tab:gray",
        "linestyle": "--",
    },
    "majority": {
        "label": "Majority",
        "color": "tab:gray",
        "linestyle": "-.",
    },
    "true": {
        "label": "True",
        "color": "tab:gray",
        "linestyle": "-",
    },
}
fig, ax = plt.subplots()
for name, roc in rocs.items():
    ax.plot(
        roc[:, 0],  # FPR
        roc[:, 1],  # TPR
        label=plot_config[name]["label"],
        color=plot_config[name]["color"],
        alpha=0.8,
        linestyle=plot_config[name]["linestyle"],
        linewidth=1,
    )
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="FPR", ylabel="TPR")
plt.legend()
ax.set_title(f"Model performances on {dims}-dimensional hypersphere data")
plt.plot(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100),
    color="tab:gray",
    alpha=0.8,
    linestyle=":",
)

plt.show()
