import numpy as np


def generate_data(N, w, n=2, minc=0, maxc=1):
    x1 = (maxc - minc) * np.random.random((N, n)) + minc
    ones = np.ones((N, 1))
    x = np.concatenate((x1, ones), axis=1)
    y = 1 * (np.dot(x, w) > 0)
    return x, y


def expert_advice(y, x, w):
    N = len(y)
    M = w.shape[0]
    x = np.hstack((x, np.ones((N, 1))))
    advice = np.zeros((N, M))
    for i, yi in enumerate(y):
        if yi == 1:
            for j in range(M):
                advice[i, j] = 1 * (
                    np.random.uniform(0, 1) <= dot_sigmoid(x[i, :], w[j, :])
                )
        else:
            for j in range(M):
                advice[i, j] = 1 * (
                    np.random.uniform(0, 1) >= dot_sigmoid(x[i, :], w[j, :])
                )
    return advice


def sigmoid(x):
    if x < -500:
        return 0
    return 1 / (1 + np.exp(-x))


# We define v = [w \gamma]
def dot_sigmoid(x, v):
    return sigmoid(np.dot(x, v))


def eval_classifier(x, advice, y, evaluator, t):
    x_1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    P = sum(y)
    N = sum(1 - y)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y)):
        if evaluator(x_1[i, :], advice[i, :]) >= t:
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
