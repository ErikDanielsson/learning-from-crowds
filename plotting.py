import numpy as np
import matplotlib.pyplot as plt
from src.utils import *


def plot_cont_datapoints(
    x, color, ax, colormap="coolwarm_r", marker="o", markersize=10
):
    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=color,
        cmap=colormap,
        marker=marker,
        s=markersize,
        vmin=0,
        vmax=1,
    )


def plot_bin_datapoints(
    x, value, ax, colors=["tab:blue", "tab:orange"], marker="o", markersize=10
):
    pos = np.array([[x[i, 0], x[i, 1]] for i in range(len(value)) if value[i] == 1])
    neg = np.array([[x[i, 0], x[i, 1]] for i in range(len(value)) if value[i] == 0])
    ax.scatter(pos[:, 0], pos[:, 1], c=colors[0], marker=marker, s=markersize)
    ax.scatter(neg[:, 0], neg[:, 1], c=colors[1], marker=marker, s=markersize)


def create_color_classifier(x, w, threshold=False):
    if threshold:
        return np.array(
            [dot_sigmoid(x[i, :], w) >= threshold for i in range(x.shape[0])]
        )
    else:
        return np.array([dot_sigmoid(x[i, :], w) for i in range(x.shape[0])])


def create_color_annotator(x, y, w, threshold=False):
    def vec_int(vec):
        return np.vectorize(int)(vec)

    conf = np.array(
        [
            dot_sigmoid(x[i, :], w) if yi == 1 else 1 - dot_sigmoid(x[i, :], w)
            for i, yi in enumerate(y)
        ]
    )
    if threshold:
        return vec_int(conf >= threshold)
    return conf


def plot_line(w, ax, color="red"):
    x_start = 0
    x_end = (
        min(1, abs(-(w[1] + w[2]) / w[0]))
        if w[0] * w[1] < 0
        else min(1, abs(-(w[2]) / w[0]))
    )
    x = np.linspace(x_start, x_end, 1000)
    ax.plot(x, -(w[0] * x + w[2]) / w[1], c=color)
