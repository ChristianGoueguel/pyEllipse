"""
**Module to compute coordinate points for Hotelling's T-squared confidence ellipses**
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, Literal

from ._utils import check_nobs, is_integer, is_number, sqrtm, tsq_limit


def hotelling_coordinates(
    x: Union[np.ndarray, pd.DataFrame],
    pcx: int = 1,
    pcy: int = 2,
    pcz: Optional[int] = None,
    conf_limit: float = 0.95,
    pts: int = 200,
    method: Literal["f", "beta"] = "f"
) -> pd.DataFrame:
    r"""
    Coordinates of Hotelling's T-squared ellipse or ellipsoid.

    Computes points on the boundary of the region where Hotelling's T-squared statistic
    of two components (ellipse) or three components (ellipsoid) equals its limit at a
    given confidence level, for plotting confidence regions on score plots.

    Parameters
    ----------
    x : numpy.ndarray or pandas.DataFrame
        Scores from PCA, PLS, ICA, or similar methods, with one row per observation and
        one column per component.
    pcx : int, default 1
        Component (1-based) on the x-axis.
    pcy : int, default 2
        Component (1-based) on the y-axis.
    pcz : int, optional
        Component (1-based) on the z-axis. When given, an ellipsoid is computed instead
        of an ellipse.
    conf_limit : float, default 0.95
        Confidence level of the ellipse, strictly between 0 and 1.
    pts : int, default 200
        Number of points: `pts` points around the ellipse, or a `pts` by `pts` grid of
        points on the ellipsoid.
    method : str, default 'f'
        Distribution of the T-squared limit, `'f'` or `'beta'`. See
        `hotelling_parameters`.

    Returns
    -------
    pandas.DataFrame
        Coordinates of the points, in columns `'x'` and `'y'`, plus `'z'` for an
        ellipsoid. The ellipsoid points are ordered as a `pts` by `pts` grid, with the
        azimuthal angle varying fastest, so each column can be reshaped to `(pts, pts)`
        for surface plots.

    Raises
    ------
    TypeError
        If `x` is not a NumPy array or a pandas DataFrame.
    ValueError
        If an argument is invalid, or if there are fewer than four (ellipse) or five
        (ellipsoid) observations.

    Notes
    -----
    With $\bar{\mathbf{x}}$ and $\mathbf{S}$ the sample mean and covariance matrix of the
    selected components and $c$ the T-squared limit, the points are
    $\bar{\mathbf{x}} + \sqrt{c}\, \mathbf{S}^{1/2} \mathbf{u}$, where $\mathbf{u}$ runs
    over the unit circle (sphere) and $\mathbf{S}^{1/2}$ is the symmetric square root of
    $\mathbf{S}$. Each point therefore has a T-squared value of exactly $c$. When the
    components are correlated, the ellipse (ellipsoid) is rotated accordingly; for
    uncorrelated scores, such as the PCA or PLS scores of the samples a model was fitted
    on, its axes are aligned with the components. Derivations are given on the Theory
    page of the documentation, <https://christiangoueguel.com/pyEllipse/theory.html>.

    Examples
    --------
    >>> import numpy as np
    >>> from pyEllipse import hotelling_coordinates
    >>> scores = np.random.default_rng(0).standard_normal((50, 3)) * [3.0, 2.0, 1.0]
    >>> ellipse = hotelling_coordinates(scores, pcx=1, pcy=2, conf_limit=0.99)
    >>> ellipse.shape
    (200, 2)
    >>> ellipsoid = hotelling_coordinates(scores, pcx=1, pcy=2, pcz=3, pts=50)
    >>> ellipsoid.shape
    (2500, 3)
    """
    if x is None:
        raise ValueError("Missing input data.")

    if isinstance(x, pd.DataFrame):
        x = x.values
    elif not isinstance(x, np.ndarray):
        raise TypeError("Input data must be a numpy array or pandas DataFrame.")

    if not is_number(conf_limit) or conf_limit <= 0 or conf_limit >= 1:
        raise ValueError("Confidence level should be a numeric value between 0 and 1.")

    if method not in ("f", "beta"):
        raise ValueError("'method' must be either 'f' or 'beta'.")

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Input data must be two-dimensional, with one column per component.")
    n, p = x.shape

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

    if not is_integer(pts) or pts <= 0:
        raise ValueError("'pts' should be a positive integer.")

    if pcz is not None:
        if not is_integer(pcz) or pcz < 1 or pcz > p:
            raise ValueError(
                f"'pcz' must be an integer between 1 and the number of components in the data ({p})."
            )

        if pcz == pcx or pcz == pcy:
            raise ValueError("'pcx', 'pcy', and 'pcz' must be different integers.")


    if pcz is None:
        result = _compute_ellipse(x, pcx, pcy, n, conf_limit, pts, method)
    else:
        result = _compute_ellipsoid(x, pcx, pcy, pcz, n, conf_limit, pts, method)
    return result


def _compute_ellipse(
    x: np.ndarray,
    pcx: int,
    pcy: int,
    n: int,
    conf_limit: float,
    pts: int,
    method: str = "f"
) -> pd.DataFrame:
    """
    Compute 2D ellipse coordinates.
    """
    theta = np.linspace(0, 2 * np.pi, pts)

    p = 2
    check_nobs(n, p)
    limit = tsq_limit(n, p, conf_limit, method)

    xy = x[:, [pcx - 1, pcy - 1]]
    unit = np.vstack([np.cos(theta), np.sin(theta)])
    coord = _map_unit(unit, xy, limit)

    return pd.DataFrame({
        'x': coord[:, 0],
        'y': coord[:, 1]
    })


def _compute_ellipsoid(
    x: np.ndarray,
    pcx: int,
    pcy: int,
    pcz: int,
    n: int,
    conf_limit: float,
    pts: int,
    method: str = "f"
) -> pd.DataFrame:
    """
    Compute 3D ellipsoid coordinates.
    """
    theta = np.linspace(0, 2 * np.pi, pts)
    phi = np.linspace(0, np.pi, pts)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    sin_phi = np.sin(phi_flat)
    cos_phi = np.cos(phi_flat)
    cos_theta = np.cos(theta_flat)
    sin_theta = np.sin(theta_flat)

    p = 3
    check_nobs(n, p)
    limit = tsq_limit(n, p, conf_limit, method)

    xyz = x[:, [pcx - 1, pcy - 1, pcz - 1]]
    unit = np.vstack([cos_theta * sin_phi, sin_theta * sin_phi, cos_phi])
    coord = _map_unit(unit, xyz, limit)

    return pd.DataFrame({
        'x': coord[:, 0],
        'y': coord[:, 1],
        'z': coord[:, 2]
    })


def _map_unit(unit: np.ndarray, scores: np.ndarray, limit: float) -> np.ndarray:
    """
    Map points of the unit circle (sphere), one per column of `unit`, onto the ellipse
    (ellipsoid) where the T-squared statistic of `scores` equals `limit`.
    """
    shape = np.sqrt(limit) * sqrtm(np.cov(scores, rowvar=False))
    # einsum rather than matmul: with Apple Accelerate, NumPy < 2 emits spurious
    # floating-point warnings from matmul
    return np.einsum("ij,jn->ni", shape, unit) + np.mean(scores, axis=0)
