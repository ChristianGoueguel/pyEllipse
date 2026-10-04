"""Shared helpers for the test suite."""
import numpy as np
from scipy import stats


def pca_scores(x: np.ndarray) -> np.ndarray:
    """PCA scores of x: uncorrelated columns, in decreasing order of variance."""
    xc = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt.T


def mahalanobis_sq(x: np.ndarray, points=None) -> np.ndarray:
    """
    Squared Mahalanobis distance of `points` (default: x itself) from the mean of x,
    using the sample covariance of x.
    """
    points = x if points is None else np.asarray(points)
    d = points - x.mean(axis=0)
    return np.einsum("ij,jk,ik->i", d, np.linalg.inv(np.cov(x, rowvar=False)), d)


def f_limit(n: int, level: float, k: int = 2) -> float:
    return k * (n - 1) / (n - k) * stats.f.ppf(level, k, n - k)


def beta_limit(n: int, level: float, k: int = 2) -> float:
    return (n - 1) ** 2 / n * stats.beta.ppf(level, k / 2, (n - k - 1) / 2)
