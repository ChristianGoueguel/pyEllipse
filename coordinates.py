"""
Coordinate Points of the Hotelling's T-squared Ellipse




This module calculates the coordinate points for drawing a Hotelling's T-squared
ellipse based on multivariate data for both 2D and 3D representations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional
from itertools import product


def ellipse_coord(
    x: Union[np.ndarray, pd.DataFrame],
    pcx: int = 1,
    pcy: int = 2,
    pcz: Optional[int] = None,
    conf_limit: float = 0.95,
    pts: int = 200
) -> pd.DataFrame:
    """
    Calculate coordinate points for drawing a Hotelling's T-squared ellipse.
    
    This function generates points for both 2D ellipses and 3D ellipsoids based on
    the Hotelling's T-squared distribution and specified components.
    
    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        Matrix or DataFrame containing scores from PCA, PLS, ICA, or other
        dimensionality reduction methods. Each column represents a component,
        and each row an observation.
    pcx : int, default=1
        Component to use for the x-axis (1-indexed).
    pcy : int, default=2
        Component to use for the y-axis (1-indexed).
    pcz : int, optional
        Component to use for the z-axis for 3D ellipsoids. If None (default),
        a 2D ellipse is computed.
    conf_limit : float, default=0.95
        Confidence level for the ellipse (between 0 and 1). Default is 0.95
        (95% confidence). Higher values result in larger ellipses.
    pts : int, default=200
        Number of points to generate for drawing the ellipse. Higher values
        result in smoother ellipses but increase computation time.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing coordinate points:
        - For 2D ellipses: columns 'x' and 'y'
        - For 3D ellipsoids: columns 'x', 'y', and 'z'
    
    Notes
    -----
    The function computes the shape and orientation of the ellipse based on the
    Hotelling's T-squared distribution. The conf_limit parameter determines the
    size of the ellipse - a higher confidence level results in a larger ellipse
    that encompasses more data points.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> 
    >>> # Generate sample data
    >>> np.random.seed(123)
    >>> data = np.random.randn(100, 5)
    >>> 
    >>> # Perform PCA
    >>> pca = PCA()
    >>> pca_scores = pca.fit_transform(data)
    >>> 
    >>> # Generate 2D ellipse coordinates
    >>> coords_2d = ellipse_coord(pca_scores, pcx=1, pcy=2)
    >>> 
    >>> # Generate 3D ellipsoid coordinates
    >>> coords_3d = ellipse_coord(pca_scores, pcx=1, pcy=2, pcz=3)
    """
    # Input validation
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
    
    # Validate component indices
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
    
    # Compute ellipse or ellipsoid coordinates
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
    
    Generates points on a 2D ellipse using parametric equations with the
    angle theta ranging from 0 to 2π.
    """
    # Generate angles for parametric representation
    theta = np.linspace(0, 2 * np.pi, pts)
    
    # Calculate T-squared limit for 2D (p=2)
    p = 2
    f_quantile = stats.f.ppf(conf_limit, p, n - p)
    tsq_limit = ((p * (n - 1)) / (n - p)) * f_quantile
    
    # Extract component columns (convert to 0-indexed)
    x_col = x[:, pcx - 1]
    y_col = x[:, pcy - 1]
    
    # Calculate means and variances
    x_mean = np.mean(x_col)
    y_mean = np.mean(y_col)
    x_var = np.var(x_col, ddof=1)
    y_var = np.var(y_col, ddof=1)
    
    # Generate ellipse coordinates using parametric equations
    x_coords = np.sqrt(tsq_limit * x_var) * np.cos(theta) + x_mean
    y_coords = np.sqrt(tsq_limit * y_var) * np.sin(theta) + y_mean
    
    return pd.DataFrame({
        'x': x_coords,
        'y': y_coords
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
    
    Generates points on a 3D ellipsoid using spherical coordinates with
    theta (azimuthal angle) ranging from 0 to 2π and phi (polar angle)
    ranging from 0 to π.
    """
    # Generate angles for parametric representation
    theta = np.linspace(0, 2 * np.pi, pts)
    phi = np.linspace(0, np.pi, pts)
    
    # Create a grid of all combinations of theta and phi
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Precompute trigonometric functions
    sin_phi = np.sin(phi_flat)
    cos_phi = np.cos(phi_flat)
    cos_theta = np.cos(theta_flat)
    sin_theta = np.sin(theta_flat)
    
    # Calculate T-squared limit for 3D (p=3)
    p = 3
    f_quantile = stats.f.ppf(conf_limit, p, n - p)
    tsq_limit = ((p * (n - 1)) / (n - p)) * f_quantile
    
    # Extract component columns (convert to 0-indexed)
    x_col = x[:, pcx - 1]
    y_col = x[:, pcy - 1]
    z_col = x[:, pcz - 1]
    
    # Calculate means and variances
    x_mean = np.mean(x_col)
    y_mean = np.mean(y_col)
    z_mean = np.mean(z_col)
    x_var = np.var(x_col, ddof=1)
    y_var = np.var(y_col, ddof=1)
    z_var = np.var(z_col, ddof=1)
    
    # Generate ellipsoid coordinates using parametric equations
    # (spherical coordinates: x = r*cos(θ)*sin(φ), y = r*sin(θ)*sin(φ), z = r*cos(φ))
    x_coords = np.sqrt(tsq_limit * x_var) * cos_theta * sin_phi + x_mean
    y_coords = np.sqrt(tsq_limit * y_var) * sin_theta * sin_phi + y_mean
    z_coords = np.sqrt(tsq_limit * z_var) * cos_phi + z_mean
    
    return pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'z': z_coords
    })


# Example usage and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    
    # Generate sample data
    np.random.seed(123)
    n_samples, n_features = 100, 10
    data = np.random.randn(n_samples, n_features)
    
    # Perform PCA (ensure we keep enough components)
    pca = PCA(n_components=min(5, n_samples, n_features))
    pca_scores = pca.fit_transform(data)
    
    print(f"PCA scores shape: {pca_scores.shape}")
    print(f"Number of components: {pca_scores.shape[1]}")
    
    # Example 1: 2D Ellipse
    print("Example 1: Generating 2D ellipse coordinates")
    coords_2d = ellipse_coord(pca_scores, pcx=1, pcy=2, conf_limit=0.95)
    print(f"Generated {len(coords_2d)} points for 2D ellipse")
    print(coords_2d.head())
    
    # Plot 2D ellipse
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.5, label='Data points')
    ax.plot(coords_2d['x'], coords_2d['y'], 'r-', linewidth=2, label='95% Confidence Ellipse')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('2D Hotelling Ellipse')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    
    # Example 2: 3D Ellipsoid
    print("\nExample 2: Generating 3D ellipsoid coordinates")
    coords_3d = ellipse_coord(pca_scores, pcx=1, pcy=2, pcz=3, conf_limit=0.95, pts=50)
    print(f"Generated {len(coords_3d)} points for 3D ellipsoid")
    print(coords_3d.head())
    
    # Plot 3D ellipsoid
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points (z-axis as third positional argument)
    ax.scatter(pca_scores[:, 0], pca_scores[:, 1], pca_scores[:, 2], 
               alpha=0.5, s=20, label='Data points')
    
    # Reshape coordinates for surface plot (pts x pts grid)
    grid_size = 50  # matches pts parameter above
    x_surf = np.array(coords_3d['x']).reshape(grid_size, grid_size)
    y_surf = np.array(coords_3d['y']).reshape(grid_size, grid_size)
    z_surf = np.array(coords_3d['z']).reshape(grid_size, grid_size)
    
    ax.plot_surface(x_surf, y_surf, z_surf,
                    alpha=0.3, color='red', edgecolor='none')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D Hotelling Ellipsoid (95% Confidence)')
    plt.tight_layout()
    
    plt.show()
    
    # Example 3: Different confidence levels
    print("\nExample 3: Comparing different confidence levels")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.5, s=20, label='Data points')
    
    for conf in [0.90, 0.95, 0.99]:
        coords = ellipse_coord(pca_scores, pcx=1, pcy=2, conf_limit=conf)
        ax.plot(coords['x'], coords['y'], linewidth=2, 
                label=f'{int(conf*100)}% Confidence')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Hotelling Ellipses at Different Confidence Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()