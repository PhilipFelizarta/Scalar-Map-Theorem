"""Data: scikit-learn's bundled 8x8 digits (1,797 images, 10 classes, no download)."""
import numpy as np


def digits_splits(seed: int = 0):
    """Fixed 60/20/20 train/validation/test split. Pixels scaled to [0, 1].

    Anything chosen by search (directions, scales) is chosen on validation; test is
    touched once, for the final report.
    """
    from sklearn.datasets import load_digits

    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    idx = np.random.default_rng(seed).permutation(len(y))
    n_tr, n_va = int(0.6 * len(y)), int(0.2 * len(y))
    tr, va, te = idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])
