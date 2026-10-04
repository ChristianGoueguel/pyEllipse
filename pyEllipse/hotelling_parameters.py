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
    r"""
    Hotelling's T-squared statistic and Hotelling's ellipse parameters.

    Computes Hotelling's T-squared statistic of each observation, its cutoffs at one or
    more confidence levels and, when two components are used, the semi-axes and rotation
    of the corresponding Hotelling's ellipse. The number of components is either fixed
    (`k`) or chosen from a cumulative explained variance `threshold`.

    Parameters
    ----------
    x : numpy.ndarray or pandas.DataFrame
        Scores from PCA, PLS, ICA, or similar methods, with one row per observation and
        one column per component.
    k : int, default 2
        Number of components to use. Ignored when `threshold` is given.
    pcx : int, default 1
        Component (1-based) on the x-axis of the ellipse when `k=2`.
    pcy : int, default 2
        Component (1-based) on the y-axis of the ellipse when `k=2`. Must differ from
        `pcx`.
    threshold : float, optional
        Cumulative explained variance threshold, in (0, 1]. When given, the smallest
        number of leading components that explain at least this proportion of the total
        variance is used (at least two).
    rel_tol : float, default 0.001
        Minimum proportion of the total variance a component must explain to be kept.
    abs_tol : float, default sys.float_info.epsilon
        Minimum variance a component must have to be kept. Must not exceed `rel_tol`.
    method : str, default 'f'
        Distribution of the T-squared cutoffs, `'f'` or `'beta'`. See Notes.
    conf_limit : float or sequence of float, default (0.95, 0.99)
        Confidence level, or levels, of the cutoffs and semi-axes, each strictly between
        0 and 1.

    Returns
    -------
    dict
        T-squared values, cutoffs, number of components and, when `k=2`, ellipse
        parameters, under the keys listed in Returned keys.

    Returned keys
    -------------
    - `'Tsquared'`: DataFrame with a column `'value'`, the T-squared statistic of each
      observation (its squared Mahalanobis distance), on the same scale as the cutoffs.
      When `k=2`, it is computed on components `pcx` and `pcy`.
    - `'cutoff_<level>pct'`: T-squared cutoff at each level of `conf_limit`, from the
      highest to the lowest level: `'cutoff_99pct'` and `'cutoff_95pct'` by default,
      `'cutoff_99.9pct'` for a level of 0.999.
    - `'nb_comp'`: number of components used.
    - `'Ellipse'` (only when `k=2`): one-row DataFrame with the semi-axes
      `'a_<level>pct'` and `'b_<level>pct'` at each level, and the rotation `'angle'` of
      the ellipse in radians. `a` is the semi-axis closest to the `pcx` direction.
      `'angle'` is 0 for uncorrelated scores, such as the PCA or PLS scores of the
      samples a model was fitted on.

    Raises
    ------
    TypeError
        If `x` is not a NumPy array or a pandas DataFrame.
    ValueError
        If an argument is invalid, if there are fewer than `k + 2` observations, or if
        fewer than two components remain after removing near-zero variance components.
    RuntimeError
        If the covariance matrix of the selected components is singular.

    Warnings
    --------
    A `UserWarning` is issued when near-zero variance components are removed, or when
    `threshold` is lower than the variance explained by the first component (two
    components are then used).

    Notes
    -----
    For observation $i$ with scores $\mathbf{x}_i$ on the $k$ selected components,
    $T^2_i = (\mathbf{x}_i - \bar{\mathbf{x}})^\top \mathbf{S}^{-1}
    (\mathbf{x}_i - \bar{\mathbf{x}})$, where $\bar{\mathbf{x}}$ and $\mathbf{S}$ are the
    sample mean and covariance matrix of the $n$ observations.

    With `method='f'` (default), the cutoff at confidence level $1 - \alpha$ is
    $\frac{k(n-1)}{n-k} F_{1-\alpha}(k, n-k)$, as in previous versions of the package.
    With `method='beta'`, it is $\frac{(n-1)^2}{n} B_{1-\alpha}(k/2, (n-k-1)/2)$, the
    exact distribution of $T^2_i$ for the observations used to estimate the mean and
    covariance (Tracy, Young and Mason, 1992), e.g. the scores of the samples a PCA or
    PLS model was built on. For these observations, the F-based cutoff is conservative,
    especially for small $n$.

    The ellipse is the set of points whose T-squared value equals the cutoff, so an
    observation lies outside it exactly when its T-squared value exceeds the cutoff.
    Derivations are given on the Theory page of the documentation,
    <https://christiangoueguel.com/pyEllipse/theory.html>.

    References
    ----------
    Tracy, N. D., Young, J. C. and Mason, R. L. (1992). Multivariate control charts for
    individual observations. *Journal of Quality Technology*, 24(2), 88-95.

    Examples
    --------
    >>> import numpy as np
    >>> from pyEllipse import hotelling_parameters
    >>> scores = np.random.default_rng(0).standard_normal((50, 3)) * [3.0, 2.0, 1.0]
    >>> res = hotelling_parameters(scores, k=2)
    >>> list(res)
    ['Tsquared', 'cutoff_99pct', 'cutoff_95pct', 'nb_comp', 'Ellipse']
    >>> round(res['cutoff_95pct'], 3)
    6.514
    >>> outliers = res['Tsquared']['value'] > res['cutoff_95pct']

    Exact cutoffs for the calibration samples, at custom confidence levels:

    >>> res = hotelling_parameters(scores, k=3, method='beta', conf_limit=(0.975, 0.999))
    >>> [key for key in res if key.startswith('cutoff')]
    ['cutoff_99.9pct', 'cutoff_97.5pct']
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
