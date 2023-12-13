from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from yan_yan_et_al import EM

# Set script parameters
random_seed = 0
np.random.seed(random_seed)
epsilon = 1e-6
expert_threshold = 0.35

# Fetch dataset
ionosphere = fetch_ucirepo(id=52)

# Data (as pandas dataframes)
X = ionosphere.data.features
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
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
folds = list(kf.split(X, y))

# Now produce the expert predictions
n_experts = n_clusters
advice = -np.ones((n, n_experts))  # Initalize to -1 to see if there is anything fishy

for e in range(n_experts):
    rands = np.random.random(n)
    advice[:, e] = [
        y[i] if c == e or r > expert_threshold else 1 - y[i]
        for i, (c, r) in enumerate(zip(cluster_assignments, rands))
    ]

# Now for each fold, run the algorithm
classifiers = np.zeros((n_folds, p + 1))  # We also have a bias (p + 1)
for i, (train_index, test_index) in enumerate(folds):
    # Assemble the data
    X_train = X[train_index, :]
    advice_train = advice[train_index, :]
    a, v, l = EM(X_train, advice_train, epsilon, 0)
    classifiers[i, :] = a
