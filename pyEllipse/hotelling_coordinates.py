"""
**Module to compute coordinate points for Hotelling's T-squared confidence ellipses**
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional
from itertools import product

def hotelling_coordinates(
    x: Union[np.ndarray, pd.DataFrame],
    pcx: int = 1,
    pcy: int = 2,
    pcz: Optional[int] = None,
    conf_limit: float = 0.95,
    pts: int = 200
) -> pd.DataFrame:
    """
    This module computes the boundary coordinate points needed to visualize Hotelling's 
    T-squared confidence regions. It supports both 2D confidence ellipses and 3D confidence 
    ellipsoids, calculating points based on the Hotelling's T-squared distribution for any 
    user-defined confidence interval.
    
    Parameters
    ----------
    *   `x` : Input matrix or data frame containing scores from PCA, PLS, ICA, or other
        dimensionality reduction methods. Each column represents a component, and each row an observation.

    *   `pcx` : Component to use for the x-axis (default=1).

    *   `pcy` : Component to use for the y-axis (default=2).
    
    *   `pcz` : Component to use for the z-axis for 3D ellipsoids. If None (default), a 2D ellipse is computed.
    
    *   `conf_limit` : Confidence level for the ellipse (between 0 and 1). Default is 0.95
        (95% confidence). Higher values result in larger ellipses.
    
    *   `pts` : Number of points to generate for drawing the ellipse. Higher values 
        result in smoother ellipses but increase computation time.
    
    Returns
    -------
    DataFrame containing coordinate points:

        - For 2D ellipses: columns 'x' and 'y'
        - For 3D ellipsoids: columns 'x', 'y', and 'z'
    """
    if x is None:
        raise ValueError("Missing input data.")
    
    if isinstance(x, pd.DataFrame):
        x = x.values
    elif not isinstance(x, np.ndarray):
        raise TypeError("Input data must be a numpy array or pandas DataFrame.")
    
    if not isinstance(conf_limit, (int, float)) or conf_limit <= 0 or conf_limit >= 1:
        raise ValueError("Confidence level should be a numeric value between 0 and 1.")
    
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    
    if not isinstance(pcx, int) or pcx < 1 or pcx > p:
        raise ValueError(f"'pcx' must be an integer between 1 and {p}.")
    
    if not isinstance(pcy, int) or pcy < 1 or pcy > p:
        raise ValueError(f"'pcy' must be an integer between 1 and {p}.")
    
    if pcx == pcy:
        raise ValueError("'pcx' and 'pcy' must be different integers.")
    
    if not isinstance(pts, int) or pts <= 0:
        raise ValueError("'pts' should be a positive integer.")
    
    if pcz is not None:
        if not isinstance(pcz, int) or pcz < 1 or pcz > p:
            raise ValueError(f"'pcz' must be an integer between 1 and {p}.")
        
        if pcz == pcx or pcz == pcy:
            raise ValueError("'pcx', 'pcy', and 'pcz' must be different integers.")
    

    if pcz is None:
        result = _compute_ellipse(x, pcx, pcy, n, conf_limit, pts)
    else:
        result = _compute_ellipsoid(x, pcx, pcy, pcz, n, conf_limit, pts)
    return result


def _compute_ellipse(
    x: np.ndarray,
    pcx: int,
    pcy: int,
    n: int,
    conf_limit: float,
    pts: int
) -> pd.DataFrame:
    """
    Compute 2D ellipse coordinates.
    """
    theta = np.linspace(0, 2 * np.pi, pts)
    
    p = 2
    f_quantile = stats.f.ppf(conf_limit, p, n - p)
    tsq_limit = ((p * (n - 1)) / (n - p)) * f_quantile
    
    x_col = x[:, pcx - 1]
    y_col = x[:, pcy - 1]
    x_mean = np.mean(x_col)
    y_mean = np.mean(y_col)
    x_var = np.var(x_col, ddof=1)
    y_var = np.var(y_col, ddof=1)        
    x_coord = np.sqrt(tsq_limit * x_var) * np.cos(theta) + x_mean
    y_coord = np.sqrt(tsq_limit * y_var) * np.sin(theta) + y_mean

    return pd.DataFrame({
        'x': x_coord,
        'y': y_coord
    })


def _compute_ellipsoid(
    x: np.ndarray,
    pcx: int,
    pcy: int,
    pcz: int,
    n: int,
    conf_limit: float,
    pts: int
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
    f_quantile = stats.f.ppf(conf_limit, p, n - p)
    tsq_limit = ((p * (n - 1)) / (n - p)) * f_quantile

    x_col = x[:, pcx - 1]
    y_col = x[:, pcy - 1]
    z_col = x[:, pcz - 1]
    x_mean = np.mean(x_col)
    y_mean = np.mean(y_col)
    z_mean = np.mean(z_col)
    x_var = np.var(x_col, ddof=1)
    y_var = np.var(y_col, ddof=1)
    z_var = np.var(z_col, ddof=1)
    x_coord = np.sqrt(tsq_limit * x_var) * cos_theta * sin_phi + x_mean
    y_coord = np.sqrt(tsq_limit * y_var) * sin_theta * sin_phi + y_mean
    z_coord = np.sqrt(tsq_limit * z_var) * cos_phi + z_mean
    
    return pd.DataFrame({
        'x': x_coord,
        'y': y_coord,
        'z': z_coord
    })
