import numpy as np
import pandas as pd
from utils import dot_sigmoid


def generate_hypersphere(dims, n_samples, filename):
    np.random.seed(0)
    sample = np.zeros((n_samples, dims))
    i = 0
    samples_left = n_samples
    while i < n_samples:
        print(i)
        pos_sample = 2 * np.random.random((dims * samples_left, dims)) - 1
        filter_sample = np.fromiter(
            (
                pos_sample[i, :]
                for i in range(pos_sample.shape[0])
                if np.linalg.norm(pos_sample[i, :]) <= 1
            ),
            dtype=np.dtype((float, dims)),
        )
        end = min(n_samples - i, filter_sample.shape[0])
        sample[i : i + end, :] = filter_sample[:end]
        i += end
    df = pd.DataFrame(sample)
    df.to_csv(filename, header=[f"d{i+1}" for i in range(dims)], index=False)


def classify_points(w, points):
    n, _ = points.shape
    # Add a bias to the matrix of points
    points_1 = np.hstack((points, np.ones((n, 1))))
    labels = 1 * (dot_sigmoid(points_1, w) >= 0.5)
    return labels


def noise_labels(labels, prob):
    rands = np.random.rand(len(labels))
    return np.fromiter(
        (x if r > prob else 1 - x for r, x in zip(rands, labels)), dtype=int
    )


def random_hyperplane(dims, orig_dist, mag):
    # Generate a random unit vector corresponding to the normal of the hyperplane
    rand_vec = np.random.randn(dims)
    rand_vec /= np.linalg.norm(rand_vec)
    # Now the distance from the origin to the hyperplane is simply given by b
    hyperplane = np.concatenate((rand_vec, [orig_dist]))
    return mag * hyperplane


def expert_classification(hplane, points, labels):
    n = points.shape[0]
    points_1 = np.hstack((points, np.ones((n, 1))))
    probs = dot_sigmoid(points_1, hplane)
    rands = np.random.rand(n)
    return np.fromiter(
        (x if r < prob else 1 - x for x, r, prob in zip(labels, rands, probs)),
        dtype=float,
    )
