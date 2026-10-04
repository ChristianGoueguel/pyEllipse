"""
**Module to compute Hotelling's T-squared statistics and parameters for confidence ellipses**
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Sequence, List, Literal
import sys
import warnings

from ._utils import (
    check_nobs,
    ellipse_axes,
    is_integer,
    is_number,
    level_label,
    tsq_limit,
)


def hotelling_parameters(
    x: Union[np.ndarray, pd.DataFrame],
    k: int = 2,
    pcx: int = 1,
    pcy: int = 2,
    threshold: Optional[float] = None,
    rel_tol: float = 0.001,
    abs_tol: float = sys.float_info.epsilon,
    method: Literal["f", "beta"] = "f",
    conf_limit: Union[float, Sequence[float]] = (0.95, 0.99)
) -> Dict:
    """
    This module provides functions to calculate Hotelling's T-squared statistics
    for multivariate data and to derive parameters for confidence ellipses based
    on Hotelling's T-squared distribution.

    Parameters
    ----------
    *   `x` : Input matrix or data frame containing scores from PCA, PLS, ICA, or similar methods. Each column represents a component, and each row an observation.

    *   `k` : Number of components to use (default=2). Ignored if threshold is provided.

    *   `pcx` : Component to use for x-axis when `k=2` (default=1).

    *   `pcy` : Component to use for y-axis when `k=2` (default=2). Must be different from `pcx`.

    *   `threshold` : Cumulative explained variance threshold (0 to 1). If provided,
        determines minimum number of components to explain at least this
        proportion of total variance.

    *   `rel_tol` : Minimum proportion of total variance a component should explain
        to be considered non-negligible (0.1% by default).

    *   `abs_tol` : Minimum absolute variance a component should have to be
        considered non-negligible (default=`sys.float_info.epsilon`).

    *   `method` : How the T-squared cutoffs are computed: `'f'` (default) or `'beta'`.
        See Notes.

    *   `conf_limit` : Confidence level, or sequence of confidence levels, each strictly
        between 0 and 1, at which the T-squared cutoffs and ellipse semi-axes are
        computed (default=`(0.95, 0.99)`). See Returns for how the results are named.

    Returns
    -------
    Dictionary containing:

        - 'Tsquared': DataFrame with the T-squared statistic for each observation (the
          squared Mahalanobis distance), on the same scale as the cutoffs. When `k=2`,
          it is computed on components `pcx` and `pcy`.
        - 'cutoff_<level>pct': T-squared cutoff at each confidence level in `conf_limit`,
          from the highest to the lowest level. With the default `conf_limit`, these are
          'cutoff_99pct' and 'cutoff_95pct'; with `conf_limit=(0.975, 0.999)`, they are
          'cutoff_99.9pct' and 'cutoff_97.5pct'.
        - 'nb_comp': Number of components retained
        - 'Ellipse': DataFrame (only when `k=2`) with the semi-axes lengths at each
          confidence level ('a_<level>pct', 'b_<level>pct', e.g. 'a_99pct', 'b_99pct',
          'a_95pct', 'b_95pct' by default) and the rotation 'angle' of the ellipse in
          radians. 'a' is the semi-axis closest to the `pcx` direction. For uncorrelated
          scores, such as the PCA or PLS scores of the samples the model was fitted on,
          'angle' is 0.

    Notes
    -----
    With `method='f'` (default), the cutoffs are k(n - 1)/(n - k) F(k, n - k), as in
    previous versions of the package. With `method='beta'`, the cutoffs follow the exact
    distribution of T-squared for the observations used to estimate the mean and
    covariance, (n - 1)^2/n Beta(k/2, (n - k - 1)/2) (Tracy, Young and Mason, 1992),
    e.g. the scores of the samples a PCA or PLS model was built on. The F-based limit is
    more conservative, especially for small `n`: it can even exceed the largest T-squared
    value any observation can reach, (n - 1)^2/n. For `n` larger than about 100, the two
    limits are close.

    When the selected components are correlated (e.g. new samples projected onto a model,
    or ICA scores), the ellipse is rotated so that it matches the T-squared statistic: an
    observation lies outside the ellipse exactly when its T-squared value exceeds the
    cutoff.

    References
    ----------
    Tracy, N. D., Young, J. C. and Mason, R. L. (1992). Multivariate control charts for
    individual observations. *Journal of Quality Technology*, 24(2), 88-95.
    """
    if x is None:
        raise ValueError("Missing input data.")

    names = None
    if isinstance(x, pd.DataFrame):
        names = [str(col) for col in x.columns]
        x = x.values
    elif not isinstance(x, np.ndarray):
        raise TypeError("Input data must be a numpy array or pandas DataFrame.")

    if not is_number(rel_tol) or rel_tol < 0:
        raise ValueError("'rel_tol' must be a non-negative numeric value.")

    if not is_number(abs_tol) or abs_tol < 0:
        raise ValueError("'abs_tol' must be a non-negative numeric value.")

    if abs_tol > rel_tol:
        raise ValueError("'abs_tol' must be less than or equal to 'rel_tol'.")

    if method not in ("f", "beta"):
        raise ValueError("'method' must be either 'f' or 'beta'.")

    levels = np.atleast_1d(np.asarray(conf_limit))
    if (
        levels.ndim != 1
        or levels.size == 0
        or levels.dtype.kind not in "iuf"
        or np.any(np.isnan(levels))
        or np.any((levels <= 0) | (levels >= 1))
    ):
        raise ValueError(
            "'conf_limit' must be a number or a sequence of numbers strictly between 0 and 1."
        )
    if len({level_label(level) for level in levels}) != levels.size:
        raise ValueError("'conf_limit' must not contain duplicated values.")

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Input data must be two-dimensional, with one column per component.")
    n, p = x.shape

    if threshold is not None:
        if not is_number(threshold) or threshold <= 0 or threshold > 1:
            raise ValueError("Threshold must be a numeric value between 0 and 1.")
    else:
        if not is_integer(k) or k < 2 or k > p:
            raise ValueError(
                f"'k' must be an integer between 2 and the number of components in the data ({p})."
            )

    if not is_integer(pcx) or pcx < 1 or pcx > p:
        raise ValueError(
            f"'pcx' must be an integer between 1 and the number of components in the data ({p})."
        )

    if not is_integer(pcy) or pcy < 1 or pcy > p:
        raise ValueError(
            f"'pcy' must be an integer between 1 and the number of components in the data ({p})."
        )

    if pcx == pcy:
        raise ValueError("'pcx' and 'pcy' must be different integers.")

    comp_var = np.var(x, axis=0, ddof=1)
    total_var = np.sum(comp_var)
    relative_var = comp_var / total_var
    nearzero_var = (relative_var < rel_tol) | (comp_var < abs_tol)

    if threshold is None:
        result = _process_fixed_comp(
            x, k, pcx, pcy, nearzero_var, relative_var, rel_tol, method, levels, names
        )
    else:
        result = _process_threshold(
            x, threshold, nearzero_var, relative_var, method, levels, names
        )
    return result


def _process_fixed_comp(
    x: np.ndarray,
    k: int,
    pcx: int,
    pcy: int,
    nearzero_var: np.ndarray,
    relative_var: np.ndarray,
    rel_tol: float,
    method: str,
    levels: np.ndarray,
    names: Optional[List[str]]
) -> Dict:
    """Process with fixed number of components."""
    result = {}
    if k == 2:
        if relative_var[pcx - 1] < rel_tol:
            raise ValueError("'pcx' has a relative variance lower than 'rel_tol'. Please check!")
        if relative_var[pcy - 1] < rel_tol:
            raise ValueError("'pcy' has a relative variance lower than 'rel_tol'. Please check!")
        # T-squared must be computed on the same components as the ellipse
        x = x[:, [pcx - 1, pcy - 1]]
    elif np.any(nearzero_var[:k]):
        warnings.warn(
            f"Components with explained variance lower than 'rel_tol' detected: "
            f"{_component_names(names, nearzero_var[:k])} removed.",
            stacklevel=3
        )
        x = x[:, ~nearzero_var]
        k = min(k, x.shape[1])
        if k < 2:
            raise ValueError(
                "Fewer than two components remain after removing near-zero variance components."
            )

    t2_values = _compute_tsquared(x, k, method, levels)
    result['Tsquared'] = t2_values['Tsq']
    result.update(_cutoffs(t2_values['Tsq_limits']))
    result['nb_comp'] = int(k)

    # Calculate ellipse parameters for 2D case
    if k == 2:
        S = np.cov(x, rowvar=False)
        semi_axes = {}
        angle = 0.0
        for label, limit in t2_values['Tsq_limits'].items():
            a, b, angle = ellipse_axes(S, limit)
            semi_axes[f'a_{label}'] = [a]
            semi_axes[f'b_{label}'] = [b]
        semi_axes['angle'] = [angle]
        result['Ellipse'] = pd.DataFrame(semi_axes)
    return result


def _process_threshold(
    x: np.ndarray,
    threshold: float,
    nearzero_var: np.ndarray,
    relative_var: np.ndarray,
    method: str,
    levels: np.ndarray,
    names: Optional[List[str]]
) -> Dict:
    """Process with cumulative variance threshold."""
    result = {}
    # Tolerance so that threshold = 1 is reachable despite floating-point rounding
    tol = np.sqrt(np.finfo(float).eps)
    cum_var = np.cumsum(relative_var)
    k_indices = np.where(cum_var >= threshold - tol)[0]

    if len(k_indices) == 0:
        raise ValueError("Threshold is too high. Cannot find enough components to meet the threshold.")

    k = int(k_indices[0]) + 1
    if k == 1:
        warnings.warn(
            f"The specified threshold ({threshold:.3f}) is lower than the variance explained "
            f"by the first component ({relative_var[0]:.3f}). The first two components (k=2) "
            f"are used.",
            stacklevel=3
        )
        k = 2

    # Check for near-zero variance components
    if np.any(nearzero_var[:k]):
        warnings.warn(
            f"Components with explained variance lower than 'rel_tol' detected within the "
            f"first {k} components to meet the threshold: "
            f"{_component_names(names, nearzero_var[:k])} removed.",
            stacklevel=3
        )
        x = x[:, ~nearzero_var]
        relative_var = relative_var[~nearzero_var]
        cum_var = np.cumsum(relative_var)
        k_indices = np.where(cum_var >= threshold - tol)[0]
        # The removed components carry negligible variance, so use all remaining
        # components if the threshold is no longer reached
        k = int(k_indices[0]) + 1 if len(k_indices) > 0 else x.shape[1]
        k = max(k, 2)
        if x.shape[1] < 2:
            raise ValueError(
                "Fewer than two components remain after removing near-zero variance components."
            )

    t2_values = _compute_tsquared(x, k, method, levels)
    result['Tsquared'] = t2_values['Tsq']
    result.update(_cutoffs(t2_values['Tsq_limits']))
    result['nb_comp'] = int(k)
    return result


def _compute_tsquared(
    x: np.ndarray,
    ncomp: int,
    method: str = "f",
    levels: Sequence[float] = (0.95, 0.99)
) -> Dict:
    """Compute Hotelling's T-squared statistic and its cutoffs."""
    n = x.shape[0]
    check_nobs(n, ncomp)
    x_subset = x[:, :ncomp]
    diff = x_subset - np.mean(x_subset, axis=0)
    cov = np.cov(x_subset, rowvar=False)

    # Squared Mahalanobis distance of each observation
    try:
        md_sq = np.sum(diff * np.linalg.solve(cov, diff.T).T, axis=1)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Error in T-squared calculation: {e}") from e

    # One limit per confidence level, from the highest to the lowest level,
    # named by level (e.g. "99pct", "95pct")
    tsq_limits = {
        f"{level_label(level)}pct": tsq_limit(n, ncomp, level, method)
        for level in sorted(levels, reverse=True)
    }
    return {
        'Tsq': pd.DataFrame({'value': md_sq}),
        'Tsq_limits': tsq_limits
    }


def _cutoffs(tsq_limits: Dict[str, float]) -> Dict[str, float]:
    """Cutoffs keyed by level, e.g. {'cutoff_99pct': ..., 'cutoff_95pct': ...}."""
    return {f'cutoff_{label}': limit for label, limit in tsq_limits.items()}


def _component_names(names: Optional[List[str]], mask: np.ndarray) -> str:
    """Names of the masked components: column names, or 1-based component numbers."""
    idx = np.flatnonzero(mask)
    if names is None:
        return ", ".join(f"component {i + 1}" for i in idx)
    return ", ".join(names[i] for i in idx)
