"""
**Module to compute coordinate points for confidence regions based on 
normal or Hotelling's T-squared distributions**
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Literal
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
    """
    This module generates coordinate points for visualizing confidence regions in multivariate data. 
    It supports both 2D confidence ellipses and 3D confidence ellipsoids at user-specified 
    confidence levels, with options for normal distribution assumptions or Hotelling's T-squared 
    distribution for small sample sizes.
    
    Parameters
    ----------
    *   `data` : Input data frame containing the variables.
    
    *   `x` : Column name for the x-axis variable.
    
    *   `y` : Column name for the y-axis variable.
    
    *   `z` : Column name for the z-axis variable (None by default).
        If provided, computes a 3D ellipsoid instead of a 2D ellipse.
    
    *   `group_by` : Column name for the grouping variable (None by default).
        This grouping variable should be categorical.
    
    *   `conf_level` : Confidence level for the ellipse/ellipsoid (between 0 and 1).
    
    *   `robust` : When `True`, uses robust estimation methods for location and scale.
        Uses sklearn's `EllipticEnvelope` for robust covariance estimation.
    
    *   `distribution` : Distribution used to calculate the quantile for the ellipse.
        Options are:

        - `'normal'`: Uses chi-square distribution (appropriate for large samples)
        - `'hotelling'`: Uses Hotelling's T² distribution (better for small samples)
    
    Returns
    -------
    DataFrame containing the coordinate points:

        - For 2D: columns 'x' and 'y'
        - For 3D: columns 'x', 'y', and 'z'
        If group_by is specified, includes the grouping column.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")
    
    if x not in data.columns:
        raise ValueError(f"Column '{x}' not found in data.")
    
    if y not in data.columns:
        raise ValueError(f"Column '{y}' not found in data.")
    
    if not isinstance(conf_level, (int, float)):
        raise TypeError("'conf_level' must be numeric.")
    
    if conf_level <= 0 or conf_level >= 1:
        raise ValueError("'conf_level' must be between 0 and 1.")
    
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
            for group_name, group_data in data.groupby(group_by):
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
            for group_name, group_data in data.groupby(group_by):
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
        except Exception as e:
            warnings.warn(f"Robust estimation failed: {e}. Using classical estimates.")
            mean_vec = np.mean(x, axis=0)
            cov_matrix = np.cov(x, rowvar=False)
    
    if np.any(np.isnan(cov_matrix)):
        raise ValueError("Covariance matrix contains NA values.")
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
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
    
    if n < 3:
        raise ValueError("At least 3 observations are required.")
    
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
        except Exception as e:
            warnings.warn(f"Robust estimation failed: {e}. Using classical estimates.")
            mean_vec = np.mean(x, axis=0)
            cov_matrix = np.cov(x, rowvar=False)
    
    if np.any(np.isnan(cov_matrix)):
        raise ValueError("Covariance matrix contains NA values.")
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
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
