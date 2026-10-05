"""
**Module to compute coordinate points for confidence regions based on 
normal or Hotelling's T-squared distributions**
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Literal
import numbers
import re
import warnings


def confidence_ellipse(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: Optional[str] = None,
    group_by: Optional[str] = None,
    conf_level: float = 0.95,
    robust: bool = False,
    distribution: Literal["normal", "hotelling"] = "normal"
) -> pd.DataFrame:
    r"""
    Coordinates of confidence ellipses or ellipsoids of raw data, optionally by group.

    Computes points on the boundary of the confidence region of two (ellipse) or three
    (ellipsoid) variables at a given confidence level, from the classical or a robust
    estimate of their mean and covariance matrix, and a normal or Hotelling's T-squared
    quantile.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the variables.
    x : str
        Column of the x-axis variable.
    y : str
        Column of the y-axis variable.
    z : str, optional
        Column of the z-axis variable. When given, an ellipsoid is computed instead of an
        ellipse.
    group_by : str, optional
        Categorical column defining groups. When given, one ellipse (ellipsoid) is
        computed for each group.
    conf_level : float, default 0.95
        Confidence level, strictly between 0 and 1.
    robust : bool, default False
        When `True`, the mean and covariance matrix are estimated robustly with
        scikit-learn's `EllipticEnvelope` (Minimum Covariance Determinant). If the robust
        fit fails, the classical estimates are used with a warning.
    distribution : str, default 'normal'
        Distribution of the quantile scaling the region: the chi-square distribution
        (`'normal'`, for large samples) or Hotelling's T-squared distribution
        (`'hotelling'`, for small samples). See Notes.

    Returns
    -------
    pandas.DataFrame
        Coordinates of the points, in columns `'x'` and `'y'`, plus `'z'` for an
        ellipsoid: 361 points around each ellipse, or a 50 by 50 grid of points on each
        ellipsoid. With `group_by`, the group of each point is in a column named after
        `group_by`.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame or `conf_level` is not a number.
    ValueError
        If a column is missing, an argument is invalid, a group has fewer than three
        observations (four for an ellipsoid), or the covariance matrix contains missing
        values.

    Notes
    -----
    With $\hat{\boldsymbol{\mu}}$ and $\hat{\boldsymbol{\Sigma}} = \mathbf{V}
    \boldsymbol{\Lambda} \mathbf{V}^\top$ the estimated mean and covariance matrix of the
    $p$ variables ($p = 2$ or 3) and $c$ the quantile, the points are
    $\hat{\boldsymbol{\mu}} + \mathbf{V} (c\boldsymbol{\Lambda})^{1/2} \mathbf{u}$, where
    $\mathbf{u}$ runs over the unit circle (sphere). The quantile is
    $c = \chi^2_{1-\alpha}(p)$ with `distribution='normal'`, and
    $c = \frac{p(n-1)}{n-p} F_{1-\alpha}(p, n-p)$ with `distribution='hotelling'`, where
    $n$ is the number of observations (of each group). Derivations are given on the
    Theory page of the documentation,
    <https://christiangoueguel.com/pyEllipse/theory.html>.

    At least $p + 1$ observations are required, so that the covariance matrix can be
    nonsingular and $n - p > 0$. A singular covariance matrix, e.g. of collinear variables,
    gives a flat region: a line segment, or a flat ellipsoid.

    The robust estimates come from `EllipticEnvelope(support_fraction=0.9,
    random_state=42)`, i.e. the reweighted Minimum Covariance Determinant estimator
    computed with the FAST-MCD algorithm (Rousseeuw and Van Driessen, 1999) on 90% of the
    observations. The reweighted covariance matrix is made consistent at the normal
    distribution (Croux and Haesbroeck, 1999), as by scikit-learn >= 1.8, so that the robust
    and classical regions agree for normal data without outliers. Older versions of
    scikit-learn do not apply this factor, so pyEllipse applies it.

    References
    ----------
    Croux, C. and Haesbroeck, G. (1999). Influence function and efficiency of the minimum
    covariance determinant scatter matrix estimator. *Journal of Multivariate Analysis*,
    71(2), 161-190.

    Rousseeuw, P. J. and Van Driessen, K. (1999). A fast algorithm for the minimum
    covariance determinant estimator. *Technometrics*, 41(3), 212-223.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyEllipse import confidence_ellipse
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.standard_normal((60, 2)), columns=['u', 'v'])
    >>> df['group'] = np.repeat(['A', 'B', 'C'], 20)
    >>> ellipses = confidence_ellipse(df, x='u', y='v', group_by='group', distribution='hotelling')
    >>> ellipses.shape
    (1083, 3)
    >>> ellipses['group'].unique().tolist()
    ['A', 'B', 'C']
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")
    
    if x not in data.columns:
        raise ValueError(f"Column '{x}' not found in data.")
    
    if y not in data.columns:
        raise ValueError(f"Column '{y}' not found in data.")
    
    if not isinstance(conf_level, numbers.Real) or isinstance(conf_level, (bool, np.bool_)):
        raise TypeError("'conf_level' must be numeric.")
    
    # Also rejects NaN
    if not 0 < conf_level < 1:
        raise ValueError("'conf_level' must be between 0 and 1.")
    conf_level = float(conf_level)
    
    if distribution not in ["normal", "hotelling"]:
        raise ValueError("'distribution' must be either 'normal' or 'hotelling'.")
    
    if z is None:
        # 2D ellipse
        if group_by is None:
            selected_data = data[[x, y]].values
            ellipse_coord = _transform_2d(selected_data, conf_level, robust, distribution)
            result = pd.DataFrame(ellipse_coord, columns=['x', 'y'])
        else:
            if group_by not in data.columns:
                raise ValueError(f"Column '{group_by}' not found in data.")
            results = []
            for group_name, group_data in data.groupby(group_by, observed=True):
                selected_data = group_data[[x, y]].values
                ellipse_coord = _transform_2d(selected_data, conf_level, robust, distribution)
                group_df = pd.DataFrame(ellipse_coord, columns=['x', 'y'])
                group_df[group_by] = group_name
                results.append(group_df)
            result = pd.concat(results, ignore_index=True)
        return result
    else:
        # 3D ellipsoid
        if z not in data.columns:
            raise ValueError(f"Column '{z}' not found in data.")
        
        if group_by is None:
            selected_data = data[[x, y, z]].values
            ellipsoid_coord = _transform_3d(selected_data, conf_level, robust, distribution)
            result = pd.DataFrame(ellipsoid_coord, columns=['x', 'y', 'z'])
        else:
            if group_by not in data.columns:
                raise ValueError(f"Column '{group_by}' not found in data.")    
            results = []
            for group_name, group_data in data.groupby(group_by, observed=True):
                selected_data = group_data[[x, y, z]].values
                ellipsoid_coord = _transform_3d(selected_data, conf_level, robust, distribution)
                group_df = pd.DataFrame(ellipsoid_coord, columns=['x', 'y', 'z'])
                group_df[group_by] = group_name
                results.append(group_df)
            result = pd.concat(results, ignore_index=True)
        return result


def _transform_2d(
    x: np.ndarray,
    conf_level: float,
    robust: bool,
    distribution: str
) -> np.ndarray:
    """
    Transform 2D data to ellipse coordinates.
    
    Parameters
    ----------
    x : np.ndarray
        2D array of shape (n_samples, 2)
    conf_level : float
        Confidence level
    robust : bool
        Whether to use robust estimation
    distribution : str
        Either 'normal' or 'hotelling'
    
    Returns
    -------
    np.ndarray
        Array of ellipse coordinates
    """
    n = x.shape[0]
    
    if n < 3:
        raise ValueError("At least 3 observations are required.")
    
    mean_vec, eigenvalues, eigenvectors = _estimate(x, robust)
    theta = np.linspace(0, 2 * np.pi, 361)
    
    if distribution == "normal":
        quantile = stats.chi2.ppf(conf_level, 2)
    else:  # hotelling
        quantile = ((2 * (n - 1)) / (n - 2)) * stats.f.ppf(conf_level, 2, n - 2)
    
    X = np.sqrt(eigenvalues[0] * quantile) * np.cos(theta)
    Y = np.sqrt(eigenvalues[1] * quantile) * np.sin(theta)
    R = np.column_stack([X, Y]) @ eigenvectors.T
    result = R + mean_vec
    return result


def _transform_3d(
    x: np.ndarray,
    conf_level: float,
    robust: bool,
    distribution: str
) -> np.ndarray:
    """
    Transform 3D data to ellipsoid coordinates.
    
    Parameters
    ----------
    x : np.ndarray
        3D array of shape (n_samples, 3)
    conf_level : float
        Confidence level
    robust : bool
        Whether to use robust estimation
    distribution : str
        Either 'normal' or 'hotelling'
    
    Returns
    -------
    np.ndarray
        Array of ellipsoid coordinates
    """
    n = x.shape[0]
    
    # With fewer than p + 1 observations, the covariance matrix is singular and n - p <= 0
    if n < 4:
        raise ValueError("At least 4 observations are required.")
    
    mean_vec, eigenvalues, eigenvectors = _estimate(x, robust)
    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, np.pi, 50)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    if distribution == "normal":
        quantile = stats.chi2.ppf(conf_level, 3)
    else:  # hotelling
        quantile = ((3 * (n - 1)) / (n - 3)) * stats.f.ppf(conf_level, 3, n - 3)
    
    X = np.sqrt(eigenvalues[0] * quantile) * np.sin(phi_flat) * np.cos(theta_flat)
    Y = np.sqrt(eigenvalues[1] * quantile) * np.sin(phi_flat) * np.sin(theta_flat)
    Z = np.sqrt(eigenvalues[2] * quantile) * np.cos(phi_flat)
    R = np.column_stack([X, Y, Z]) @ eigenvectors.T
    result = R + mean_vec
    return result


def _estimate(x: np.ndarray, robust: bool):
    """
    Mean vector of the rows of x, and eigenvalues (in increasing order) and eigenvectors of
    their covariance matrix, both classical or robust.
    """
    if not robust:
        mean_vec = np.mean(x, axis=0)
        cov_matrix = np.cov(x, rowvar=False)
    else:
        from sklearn.covariance import EllipticEnvelope
        try:
            robust_cov = EllipticEnvelope(support_fraction=0.9, random_state=42)
            robust_cov.fit(x)
            mean_vec = robust_cov.location_
            cov_matrix = robust_cov.covariance_
            if _sklearn_version() < (1, 8):
                # The reweighted covariance matrix is computed from the observations within
                # the 97.5% chi-square quantile of the raw MCD distances. scikit-learn < 1.8
                # does not rescale it, so it is too small at the normal distribution (by
                # about 10% for p = 2): apply the factor of scikit-learn >= 1.8 (Croux and
                # Haesbroeck, 1999).
                p = x.shape[1]
                cov_matrix = cov_matrix * 0.975 / stats.chi2.cdf(stats.chi2.ppf(0.975, p), p + 2)
        except Exception as e:
            warnings.warn(f"Robust estimation failed: {e}. Using classical estimates.")
            mean_vec = np.mean(x, axis=0)
            cov_matrix = np.cov(x, rowvar=False)
    
    if np.any(np.isnan(cov_matrix)):
        raise ValueError("Covariance matrix contains NA values.")
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # The covariance matrix of collinear variables is singular, and rounding can make its
    # smallest eigenvalues slightly negative: clip them, so that the region is flat
    return mean_vec, np.clip(eigenvalues, 0, None), eigenvectors


def _sklearn_version() -> tuple:
    """(major, minor) version of scikit-learn."""
    import sklearn
    return tuple(int(v) for v in re.match(r"(\d+)\.(\d+)", sklearn.__version__).groups())
