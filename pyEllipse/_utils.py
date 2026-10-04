"""
Helpers shared by `hotelling_parameters` and `hotelling_coordinates`.
"""
import numbers

import numpy as np
from scipy import stats


def is_integer(x) -> bool:
    """True for an integer scalar (Python or NumPy), excluding booleans."""
    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def is_number(x) -> bool:
    """True for a real scalar (Python or NumPy) that is not NaN, excluding booleans."""
    return (
        isinstance(x, numbers.Real)
        and not isinstance(x, (bool, np.bool_))
        and not np.isnan(x)
    )


def tsq_limit(n: int, k: int, conf_limit: float, method: str) -> float:
    """
    Upper control limit of Hotelling's T-squared for n observations and k components.

    "beta": exact limit for the observations used to estimate the mean and
    covariance, e.g. the PCA scores themselves (Tracy, Young & Mason, 1992).
    "f": k(n - 1)/(n - k) F-quantile, as used in pyEllipse <= 0.1.5.
    """
    if method == "beta":
        return float(((n - 1) ** 2 / n) * stats.beta.ppf(conf_limit, k / 2, (n - k - 1) / 2))
    return float((k * (n - 1) / (n - k)) * stats.f.ppf(conf_limit, k, n - k))


def level_label(level: float) -> str:
    """Confidence level as a percentage label: 0.95 -> "95", 0.975 -> "97.5", 0.999 -> "99.9"."""
    return f"{100 * level:.10g}"


def check_nobs(n: int, k: int) -> None:
    if n < k + 2:
        raise ValueError(
            f"At least {k + 2} observations are needed to use {k} components (got {n})."
        )


def is_uncorrelated(S: np.ndarray) -> bool:
    """
    True when the columns behind covariance matrix S are uncorrelated up to rounding,
    as for PCA scores and PLS X-scores of the samples the model was fitted on.
    """
    sd = np.sqrt(np.diag(S))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = S / np.outer(sd, sd)
    # A zero-variance column has zero covariances, i.e. it is uncorrelated
    r = np.nan_to_num(r[np.triu_indices_from(r, k=1)])
    return bool(np.all(np.abs(r) < np.sqrt(np.finfo(float).eps)))


def ellipse_axes(S: np.ndarray, tsq_limit: float):
    """
    Semi-axes and rotation angle of the 2D ellipse {u : (u - m)' S^-1 (u - m) = tsq_limit}.

    Returns (a, b, angle), where `a` is the semi-axis closest to the x-axis. Uncorrelated
    scores give exactly the axis-aligned ellipse of pyEllipse <= 0.1.5: angle = 0,
    a = sqrt(tsq_limit * var(x)) and b = sqrt(tsq_limit * var(y)). This also avoids an
    arbitrary angle when the ellipse is a circle (equal variances, e.g. SIMPLS scores).
    """
    if is_uncorrelated(S):
        return np.sqrt(tsq_limit * S[0, 0]), np.sqrt(tsq_limit * S[1, 1]), 0.0
    values, vectors = np.linalg.eigh(S)
    # Decreasing order, as R's eigen()
    values, vectors = values[::-1], vectors[:, ::-1]
    i = 0 if abs(vectors[0, 0]) >= abs(vectors[0, 1]) else 1
    v = vectors[:, i]
    if v[0] < 0:
        v = -v
    return (
        np.sqrt(tsq_limit * values[i]),
        np.sqrt(tsq_limit * values[1 - i]),
        float(np.arctan2(v[1], v[0])),
    )


def sqrtm(S: np.ndarray) -> np.ndarray:
    """
    Symmetric square root of a covariance matrix. It maps the unit circle (sphere)
    onto the ellipse (ellipsoid) with shape S; for uncorrelated scores it is diag(sd),
    which gives the axis-aligned ellipse of pyEllipse <= 0.1.5.
    """
    if is_uncorrelated(S):
        return np.diag(np.sqrt(np.diag(S)))
    values, vectors = np.linalg.eigh(S)
    return vectors @ np.diag(np.sqrt(np.clip(values, 0, None))) @ vectors.T
