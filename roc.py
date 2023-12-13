import random
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from yan_yan_et_al import *

np.random.seed(1234)


def eval_classifier(x, y, w):
    P = sum(y)
    N = sum(1 - y)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y)):
        if np.dot(x[i, :], w) > 0:
            if y[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y[i] == 1:
                FN += 1
            else:
                TN += 1
    # prec = TP / (TP + FP)
    # rec = TP / (TP + FN)
    # F_score = 2 * prec * rec / (prec + rec)
    return TP / P, FP / N


a_real = np.array([1, -1, 0])
x, y = generate_data(100, a_real)
x = x[:, 0:-1]
v_real = np.array(
    [
        [3, -1, 0],
        [1, 1, -1],
        [2, 2, -1],
        [-2, -2, 1],
    ]
)


gammas = np.linspace(0.01, 10, 50)
evals = []
advice = expert_advice(y, x, v_real)
x_1 = np.column_stack((x, np.ones(x.shape[0])))

for gamma in gammas:
    print(gamma)
    np.random.seed(1234)

    a, v, l = EM(x, advice, 1e-1, gamma)
    evals.append(eval_classifier(x_1, y, a))
    print(l[-1])

evals = np.array(evals)
plt.scatter(evals[:, 1], evals[:, 0])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
print(evals)
print(y)
