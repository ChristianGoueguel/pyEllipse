"""
Hotelling's T-squared Statistic and Ellipse Parameters



This module calculates Hotelling's T-squared statistic and, when applicable,
the lengths of the semi-axes of the Hotelling's ellipse.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional, Dict
import sys


def ellipse_param(
    x: Union[np.ndarray, pd.DataFrame],
    k: int = 2,
    pcx: int = 1,
    pcy: int = 2,
    threshold: Optional[float] = None,
    rel_tol: float = 0.001,
    abs_tol: float = sys.float_info.epsilon
) -> Dict:
    """
    Calculate Hotelling's T-squared statistic and ellipse parameters.
    
    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        Matrix or DataFrame containing scores from PCA, PLS, ICA, or similar methods.
        Each column represents a component, and each row an observation.
    k : int, default=2
        Number of components to use. Ignored if threshold is provided.
    pcx : int, default=1
        Component to use for x-axis when k=2 (1-indexed).
    pcy : int, default=2
        Component to use for y-axis when k=2 (1-indexed).
    threshold : float, optional
        Cumulative explained variance threshold (0 to 1). If provided,
        determines minimum number of components to explain at least this
        proportion of total variance.
    rel_tol : float, default=0.001
        Minimum proportion of total variance a component should explain
        to be considered non-negligible (0.1% by default).
    abs_tol : float, default=sys.float_info.epsilon
        Minimum absolute variance a component should have to be
        considered non-negligible.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'Tsquare': DataFrame with T-squared statistic for each observation
        - 'Ellipse': DataFrame with semi-axes lengths (only when k=2)
        - 'cutoff_99pct': T-squared cutoff at 99% confidence
        - 'cutoff_95pct': T-squared cutoff at 95% confidence
        - 'nb_comp': Number of components used
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> 
    >>> # Generate sample data
    >>> np.random.seed(123)
    >>> data = np.random.randn(100, 10)
    >>> 
    >>> # Perform PCA
    >>> pca = PCA()
    >>> pca_scores = pca.fit_transform(data)
    >>> 
    >>> # Calculate T-squared with fixed components
    >>> result = ellipse_param(pca_scores, k=2)
    >>> 
    >>> # Calculate with variance threshold
    >>> result = ellipse_param(pca_scores, threshold=0.95)
    """
    # Input validation
    if x is None:
        raise ValueError("Missing input data.")
    
    if isinstance(x, pd.DataFrame):
        x = x.values
    elif not isinstance(x, np.ndarray):
        raise TypeError("Input data must be a numpy array or pandas DataFrame.")
    
    if not isinstance(rel_tol, (int, float)) or rel_tol < 0:
        raise ValueError("'rel_tol' must be a non-negative numeric value.")
    
    if not isinstance(abs_tol, (int, float)) or abs_tol < 0:
        raise ValueError("'abs_tol' must be a non-negative numeric value.")
    
    if abs_tol > rel_tol:
        raise ValueError("'abs_tol' must be less than or equal to 'rel_tol'.")
    
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    
    # Validate threshold
    if threshold is not None:
        if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold > 1:
            raise ValueError("Threshold must be a numeric value between 0 and 1.")
    else:
        if not isinstance(k, int) or k < 2 or k > p:
            raise ValueError(f"'k' must be an integer between 2 and {p}.")
    
    # Validate pcx and pcy
    if not isinstance(pcx, int) or pcx < 1 or pcx > p:
        raise ValueError(f"'pcx' must be an integer between 1 and {p}.")
    
    if not isinstance(pcy, int) or pcy < 1 or pcy > p:
        raise ValueError(f"'pcy' must be an integer between 1 and {p}.")
    
    if pcx == pcy:
        raise ValueError("'pcx' and 'pcy' must be different integers.")
    
    # Calculate component variances
    comp_var = np.var(x, axis=0, ddof=1)
    total_var = np.sum(comp_var)
    relative_var = comp_var / total_var
    nearzero_var = (relative_var < rel_tol) | (comp_var < abs_tol)
    
    # Process based on threshold or fixed components
    if threshold is None:
        result = _process_fixed_comp(
            x, k, pcx, pcy, nearzero_var, comp_var, relative_var, rel_tol
        )
    else:
        result = _process_threshold(
            x, threshold, nearzero_var, relative_var
        )
    
    return result


def _process_fixed_comp(
    x: np.ndarray,
    k: int,
    pcx: int,
    pcy: int,
    nearzero_var: np.ndarray,
    comp_var: np.ndarray,
    relative_var: np.ndarray,
    rel_tol: float
) -> Dict:
    """Process with fixed number of components."""
    result = {}
    
    # Check for near-zero variance components
    if np.any(nearzero_var[:k]):
        removed_idx = np.where(nearzero_var[:k])[0]
        print(f"Warning: Components with explained variance lower than 'rel_tol' "
              f"detected: {removed_idx.tolist()} removed.")
        x = x[:, ~nearzero_var]
        k = min(k, x.shape[1])
    
    # Compute T-squared
    try:
        t2_values = _compute_tsquared(x, k)
    except Exception as e:
        raise RuntimeError(f"Error in T-squared calculation: {str(e)}")
    
    result['Tsquare'] = t2_values['Tsq']
    result['cutoff_99pct'] = t2_values['Tsq_limit1']
    result['cutoff_95pct'] = t2_values['Tsq_limit2']
    result['nb_comp'] = k
    
    # Calculate ellipse parameters for 2D case
    if k == 2:
        # Convert to 0-indexed
        pcx_idx = pcx - 1
        pcy_idx = pcy - 1
        
        if relative_var[pcx_idx] < rel_tol:
            raise ValueError(
                "'pcx' has a relative variance lower than 'rel_tol'. Please check!"
            )
        if relative_var[pcy_idx] < rel_tol:
            raise ValueError(
                "'pcy' has a relative variance lower than 'rel_tol'. Please check!"
            )
        
        result['Ellipse'] = pd.DataFrame({
            'a_99pct': [np.sqrt(t2_values['Tsq_limit1'] * comp_var[pcx_idx])],
            'b_99pct': [np.sqrt(t2_values['Tsq_limit1'] * comp_var[pcy_idx])],
            'a_95pct': [np.sqrt(t2_values['Tsq_limit2'] * comp_var[pcx_idx])],
            'b_95pct': [np.sqrt(t2_values['Tsq_limit2'] * comp_var[pcy_idx])]
        })
    
    return result


def _process_threshold(
    x: np.ndarray,
    threshold: float,
    nearzero_var: np.ndarray,
    relative_var: np.ndarray
) -> Dict:
    """Process with cumulative variance threshold."""
    result = {}
    
    # Find number of components needed for threshold
    cum_var = np.cumsum(relative_var)
    k_indices = np.where(cum_var >= threshold)[0]
    
    if len(k_indices) == 0:
        raise ValueError(
            "Threshold is too high. Cannot find enough components to meet the threshold."
        )
    
    k = k_indices[0] + 1  # Convert to 1-indexed count
    
    if k == 1:
        print(f"Warning: The specified threshold ({threshold:.3f}) is lower than "
              f"the variance explained by the first component ({relative_var[0]:.3f}). "
              f"Using the first two components (k=2).")
        k = 2
    
    # Check for near-zero variance components
    if np.any(nearzero_var[:k]):
        removed_idx = np.where(nearzero_var[:k])[0]
        print(f"Warning: Components with explained variance lower than 'rel_tol' "
              f"detected within the first {k} components: {removed_idx.tolist()} removed.")
        x = x[:, ~nearzero_var]
        relative_var = relative_var[~nearzero_var]
        cum_var = np.cumsum(relative_var)
        k_indices = np.where(cum_var >= threshold)[0]
        k = k_indices[0] + 1 if len(k_indices) > 0 else x.shape[1]
    
    # Compute T-squared
    try:
        t2_values = _compute_tsquared(x, k)
    except Exception as e:
        raise RuntimeError(f"Error in T-squared calculation: {str(e)}")
    
    result['Tsquare'] = t2_values['Tsq']
    result['cutoff_99pct'] = t2_values['Tsq_limit1']
    result['cutoff_95pct'] = t2_values['Tsq_limit2']
    result['nb_comp'] = k
    
    return result


def _compute_tsquared(x: np.ndarray, ncomp: int) -> Dict:
    """Compute Hotelling's T-squared statistic."""
    n = x.shape[0]
    x_subset = x[:, :ncomp]
    
    # Calculate Mahalanobis distance
    mean = np.mean(x_subset, axis=0)
    cov = np.cov(x_subset, rowvar=False)
    
    # Compute Mahalanobis distance for each observation
    diff = x_subset - mean
    inv_cov = np.linalg.inv(cov)
    md_sq = np.sum(diff @ inv_cov * diff, axis=1)
    
    # Calculate cutoff values using F-distribution
    f_99 = stats.f.ppf(0.99, ncomp, n - ncomp)
    f_95 = stats.f.ppf(0.95, ncomp, n - ncomp)
    
    tsq_limit1 = (ncomp * (n - 1) / (n - ncomp)) * f_99
    tsq_limit2 = (ncomp * (n - 1) / (n - ncomp)) * f_95
    
    # Calculate T-squared values
    tsq_values = ((n - ncomp) / (ncomp * (n - 1))) * md_sq
    
    tsq_df = pd.DataFrame({'value': tsq_values})
    
    return {
        'Tsq': tsq_df,
        'Tsq_limit1': tsq_limit1,
        'Tsq_limit2': tsq_limit2
    }


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(123)
    n_samples, n_features = 100, 10
    data = np.random.randn(n_samples, n_features)
    
    # Simulate PCA scores (for demonstration)
    from sklearn.decomposition import PCA
    pca = PCA()
    pca_scores = pca.fit_transform(data)
    
    # Example 1: Fixed number of components
    print("Example 1: Using 2 components")
    result1 = ellipse_param(pca_scores, k=2)
    print(f"Number of components: {result1['nb_comp']}")
    print(f"99% cutoff: {result1['cutoff_99pct']:.4f}")
    print(f"95% cutoff: {result1['cutoff_95pct']:.4f}")
    if 'Ellipse' in result1:
        print("Ellipse parameters:")
        print(result1['Ellipse'])
    print()
    
    # Example 2: Using variance threshold
    print("Example 2: Using 95% variance threshold")
    result2 = ellipse_param(pca_scores, threshold=0.95)
    print(f"Number of components: {result2['nb_comp']}")
    print(f"99% cutoff: {result2['cutoff_99pct']:.4f}")
    print(f"95% cutoff: {result2['cutoff_95pct']:.4f}")