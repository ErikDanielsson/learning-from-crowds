from yan_yan_et_al import yan_yan_et_al, real_likelihood
import numpy as np
from sklearn.model_selection import KFold


def optimize_reg_yan_yan(X, advice, ls, epsilon):
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=False)  # , random_state=random_seed)
    folds = list(kf.split(X))
    likelihoods = np.zeros(len(ls))
    for i, (train_index, test_index) in enumerate(folds):
        X_train = X[train_index, :]
        advice_train = advice[train_index, :]
        X_test = X[test_index, :]
        advice_test = advice[test_index, :]
        N, T = advice_test.shape
        X_test = np.hstack((X_test, np.ones((N, 1))))
        for j, l in enumerate(ls):
            a, v, _ = yan_yan_et_al(X_train, advice_train, epsilon, l)
            likelihoods[j] += real_likelihood(a, v, X_test, advice_test, N, T)
    best_ind = np.argmax(likelihoods)
    l_opt = ls[best_ind]
    a, v, _ = yan_yan_et_al(X, advice, epsilon, l_opt)
    return a, v

def optimize_tree(X, advice, epsilon):