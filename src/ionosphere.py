from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from src.yan_yan_et_al import yan_yan_et_al

# from raykar_logistic import raykar_et_al
from majority import majority, true_classifier, concat
from src.utils import eval_classifier, dot_sigmoid
from plotting import create_color_annotator
import umap
from math import isnan

# Set script parameters
random_seed = 0
np.random.seed(random_seed)
epsilon = 1
expert_threshold = 0.35
n_roc_samples = 100000

# Fetch dataset
ionosphere = fetch_ucirepo(id=52)

# Data (as pandas dataframes)
X = ionosphere.data.features
# print(X)
y = ionosphere.data.targets
# combined = X.join(y)
# combined = combined.dropna()
# y = combined.num
# X = combined.drop(columns=["num"])
# print(X)
X = np.array(X)
X_1 = np.hstack((X, np.ones((X.shape[0], 1))))
print(X.shape)
n, p = X.shape
print(y)

# Map the y variables to the labels (0 or 1)
d = {"g": 1, "b": 0}
y = np.array([d[yi] for yi in y.Class])
print(y)
# Produce the expert clusters on the full dataset
n_clusters = 5
kmeans = KMeans(
    n_clusters=n_clusters,
    n_init="auto",
    init="random",
    random_state=random_seed,
).fit(X)

# Attribute each datapoint in the dataset to its cluster
cluster_assignments = kmeans.predict(X)

reducer = umap.UMAP()
embedding = reducer.fit_transform(X)

# Create the 5 folds
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=False)  # , random_state=random_seed)
folds = list(kf.split(X))

# Now produce the expert predictions
n_experts = n_clusters
advice = -np.ones((n, n_experts))  # Initalize to -1 to see if there is anything fishy

for e in range(n_experts):
    rands = np.random.random(n)
    advice[:, e] = [
        y[i] if c == e or r > expert_threshold else 1 - y[i]
        for i, (c, r) in enumerate(zip(cluster_assignments, rands))
    ]
print(advice)

# Display the stats for different experts
for e in range(n_experts):
    diff = np.zeros(n_clusters)
    cluster_size = np.zeros(n_clusters)
    for j, c in enumerate(cluster_assignments):
        diff[c] += abs(advice[j, e] - y[j])
        cluster_size[c] += 1
    print(np.divide(diff, cluster_size))
plt.figure()
for i in range(n_experts):
    plt.subplot(2, 3, i + 1)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[abs(y[j] - advice[j, i]) for j in range(len(y))],
    )
plt.subplot(2, 3, 6)
plt.scatter(embedding[:, 0], embedding[:, 1], c=y)

# Now for each fold, run the algorithm
classifiers = np.zeros((n_folds, p + 1))  # We also have a bias (p + 1)
classification_thresholds = np.linspace(0, 1, n_roc_samples)
roc_EM_algorithm = np.zeros((n_folds, n_roc_samples, 2))
roc_majority = np.zeros((n_folds, n_roc_samples, 2))
roc_true = np.zeros((n_folds, n_roc_samples, 2))
roc_concat = np.zeros((n_folds, n_roc_samples, 2))
for i, (train_index, test_index) in enumerate(folds):
    print(f"Fold {i}")
    # Assemble the data
    X_train = X[train_index]
    advice_train = advice[train_index]
    y_train = y[train_index]
    # a, v, l = yan_yan_et_al(X_train, advice_train, epsilon, 0)
    w_concat = concat(X_train, advice_train, 0)
    w_true = true_classifier(X_train, y_train, 0)
    # a, v, l = yan_yan_et_al(X_train, advice_train, epsilon, 0)
    w_majority = majority(X_train, advice_train, 0)

    # classifiers[i, :] = a
    y_test = y[test_index]
    X_test = X[test_index]
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
    advice_test = advice[test_index]
    yan_yan_votes_logistically = np.zeros(len(y_test))
    majority_votes = np.zeros(len(y_test))
    expert_opinion = np.zeros((n))

    # yan_yan_votes_logistically = dot_sigmoid(X_test, a)
    majority_votes = dot_sigmoid(X_test, w_majority)
    true_votes = dot_sigmoid(X_test, w_true)
    concat_votes = dot_sigmoid(X_test, w_concat)
    # for j in range(len(y_test)):
    #     posteriors[j] = # posterior(X_test[j, :], advice_test[j, :], a, v)
    #     majority_votes[j] = majority(X_test[j, :], advice_test[j, :])
    for j, t in enumerate(classification_thresholds):
        # TPR, FPR = eval_classifier(y_test, yan_yan_votes_logistically, t)
        # roc_EM_algorithm[i, j, :] += (FPR, TPR)
        TPR, FPR = eval_classifier(y_test, majority_votes, t)
        roc_majority[i, j, :] += (FPR, TPR)
        TPR, FPR = eval_classifier(y_test, true_votes, t)
        roc_true[i, j, :] += (FPR, TPR)
        TPR, FPR = eval_classifier(y_test, concat_votes, t)
        roc_concat[i, j, :] += (FPR, TPR)
    """"
    plt.figure()
    for j in range(n_experts):
        conf = create_color_annotator(X_1, y, v[j, :])
        plt.subplot(2, 3, j + 1)
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=conf,
            cmap="coolwarm_r",
        )
    """

plt.show()
# Now plot the roc curve
for i in range(n_folds):
    plt.subplot(2, 3, i + 1)
    plt.plot(roc_EM_algorithm[i, :, 0], roc_EM_algorithm[i, :, 1], label="EM")
    plt.plot(roc_majority[i, :, 0], roc_majority[i, :, 1], label="Majority")
    plt.plot(roc_true[i, :, 0], roc_true[i, :, 1], label="True")
    plt.plot(roc_concat[i, :, 0], roc_concat[i, :, 1], label="Concat")

plt.subplot(2, 3, 6)
roc_EM_algorithm = np.mean(roc_EM_algorithm, axis=0)
roc_majority = np.mean(roc_majority, axis=0)
roc_true = np.mean(roc_true, axis=0)
roc_concat = np.mean(roc_concat, axis=0)
plt.plot(roc_EM_algorithm[:, 0], roc_EM_algorithm[:, 1], label="EM")
plt.plot(roc_majority[:, 0], roc_majority[:, 1], label="Majority")
plt.plot(roc_true[:, 0], roc_true[:, 1], label="True")
plt.plot(roc_concat[:, 0], roc_concat[:, 1], label="Concat")
plt.legend()
plt.show()
