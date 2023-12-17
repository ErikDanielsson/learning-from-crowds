from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from yan_yan_et_al import yan_yan_et_al, posterior
from raykar_logistic import raykar_et_al
from majority import majority
from utils import eval_classifier, dot_sigmoid

# Set script parameters
random_seed = 123
np.random.seed(random_seed)
epsilon = 1e-3
expert_threshold = 0.35
n_roc_samples = 100000

# Fetch dataset
ionosphere = fetch_ucirepo(id=52)

# Data (as pandas dataframes)
X = ionosphere.data.features
X = np.array(X)
n, p = X.shape
y = ionosphere.data.targets

# Map the y variables to the labels (0 or 1)
d = {"g": 1, "b": 0}
y = np.array([d[yi] for yi in y.Class])

# Produce the expert clusters on the full dataset
n_clusters = 5
kmeans = KMeans(
    n_clusters=n_clusters,
    n_init="auto",
    random_state=random_seed,
).fit(X)

# Attribute each datapoint in the dataset to its cluster
cluster_assignments = kmeans.predict(X)

# Create the 5 folds
n_folds = 4
kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
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

# Display the stats for different experts
for e in range(n_experts):
    diff = np.zeros(n_clusters)
    cluster_size = np.zeros(n_clusters)
    for j, c in enumerate(cluster_assignments):
        diff[c] += abs(advice[j, e] - y[j])
        cluster_size[c] += 1
    print(e, np.divide(diff, cluster_size))

# Now for each fold, run the algorithm
classifiers = np.zeros((n_folds, p + 1))  # We also have a bias (p + 1)
classification_thresholds = np.linspace(0, 1, n_roc_samples)
roc_EM_algorithm = np.zeros((n_folds, n_roc_samples, 2))
roc_majority = np.zeros((n_folds, n_roc_samples, 2))
for i, (train_index, test_index) in enumerate(folds):
    print(f"Fold {i}")
    # Assemble the data
    X_train = X[train_index]
    advice_train = advice[train_index]
    # a, v, l = yan_yan_et_al(X_train, advice_train, epsilon, 0)
    a, v, l = yan_yan_et_al(X_train, advice_train, epsilon, 0)
    w_majority = majority(X_train, advice_train, 0)

    classifiers[i, :] = a
    y_test = y[test_index]
    X_test = X[test_index]
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
    advice_test = advice[test_index]
    posteriors = np.zeros(len(y_test))
    majority_votes = np.zeros(len(y_test))
    expert_opinion = np.zeros((n))
    posteriors[:] = dot_sigmoid(X_test, a)
    majority_votes[:] = dot_sigmoid(X_test, w_majority)
    # for j in range(len(y_test)):
    #     posteriors[j] = # posterior(X_test[j, :], advice_test[j, :], a, v)
    #     majority_votes[j] = majority(X_test[j, :], advice_test[j, :])
    for j, t in enumerate(classification_thresholds):
        TPR, FPR = eval_classifier(X_test, advice_test, y_test, posteriors, t)
        roc_EM_algorithm[i, j, :] += (FPR, TPR)
        TPR, FPR = eval_classifier(X_test, advice_test, y_test, majority_votes, t)
        roc_majority[i, j, :] += (FPR, TPR)


# Now plot the roc curve
for i in range(n_folds):
    plt.subplot(2, 2, i + 1)
    plt.plot(roc_EM_algorithm[i, :, 0], roc_EM_algorithm[i, :, 1], label="EM")
    plt.plot(roc_majority[i, :, 0], roc_majority[i, :, 1], label="Majority")
plt.legend()
plt.show()
