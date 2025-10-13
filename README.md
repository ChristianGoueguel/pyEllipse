# pyEllipse

A Python package for computing Hotelling's T² statistics and generating confidence ellipse/ellipsoid coordinates for multivariate data analysis and visualization.

## Overview

`pyEllipse` provides three main functions for analyzing multivariate data:

1. **`hotelling_parameters`** - Calculate Hotelling's T² statistics and ellipse parameters
2. **`hotelling_coordinates`** - Generate Hotelling's ellipse/ellipsoid coordinates from PCA/PLS scores
3. **`confidence_ellipse`** - Compute confidence ellipse/ellipsoid coordinates from raw data with grouping support

## Installation

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

Then install the package:
```bash
pip install pyEllipse
```

## Functions

### 1. `hotelling_parameters` - Hotelling's T² Statistics

Calculate Hotelling's T² statistic and ellipse parameters from component scores (PCA, PLS, ICA, etc.).

**Key Features:**
- Computes T² statistic for outlier detection
- Provides 95% and 99% confidence cutoffs
- Calculates ellipse semi-axes for 2D plots
- Supports automatic component selection via variance threshold

**Parameters:**
- `x`: Matrix/DataFrame of component scores
- `k`: Number of components (default: 2)
- `pcx`, `pcy`: Components for x/y axes (default: 1, 2)
- `threshold`: Cumulative variance threshold for automatic component selection
- `rel_tol`, `abs_tol`: Variance thresholds for component filtering

**Returns:**
- `Tsquared`: DataFrame with T² values for each observation
- `cutoff_99pct`, `cutoff_95pct`: Confidence cutoffs
- `Ellipse`: Semi-axes lengths (when k=2)
- `nb_comp`: Number of components used

### 2. `hotelling_coordinates` - Hotelling's Ellipse Coordinates

Generate coordinate points for drawing Hotelling's T² ellipses/ellipsoids from component scores.

**Key Features:**
- Creates smooth ellipse boundaries for plotting
- Supports both 2D ellipses and 3D ellipsoids
- Uses Hotelling's T² distribution for confidence regions
- Customizable number of points for smooth curves

**Parameters:**
- `x`: Matrix/DataFrame of component scores
- `pcx`, `pcy`, `pcz`: Component indices for axes
- `conf_limit`: Confidence level (default: 0.95)
- `pts`: Number of points to generate (default: 200)

**Returns:**
- DataFrame with 'x', 'y' columns (2D) or 'x', 'y', 'z' columns (3D)

### 3. `confidence_ellipse` - Confidence Ellipse from Raw Data

Compute confidence ellipse/ellipsoid coordinates directly from raw data with support for grouping.

**Key Features:**
- Works with raw data (not component scores)
- Supports grouping by categorical variables
- Choice of 'normal' (chi-square) or 'hotelling' (T²) distributions
- Optional robust estimation for outlier resistance
- Unified API for 2D and 3D (via optional `z` parameter)

**Parameters:**
- `data`: DataFrame containing variables
- `x`, `y`, `z`: Column names for axes (z is optional)
- `group_by`: Column name for grouping
- `conf_level`: Confidence level (default: 0.95)
- `robust`: Use robust estimation (default: False)
- `distribution`: 'normal' or 'hotelling' (default: 'normal')

**Returns:**
- DataFrame with coordinate points and optional grouping column

## Usage Examples

### Example 1: Hotelling's T² from PCA Scores

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pyEllipse import hotelling_parameters, hotelling_coordinates

# Generate sample data
np.random.seed(42)
data = np.random.randn(100, 10)

# Perform PCA
pca = PCA()
pca_scores = pca.fit_transform(data)

# Calculate T² statistics
results = hotelling_parameters(pca_scores, k=2)
print(f"95% cutoff: {results['cutoff_95pct']:.3f}")
print(f"99% cutoff: {results['cutoff_99pct']:.3f}")

# Generate ellipse coordinates for plotting
ellipse_95 = hotelling_coordinates(pca_scores, pcx=1, pcy=2, conf_limit=0.95)
ellipse_99 = hotelling_coordinates(pca_scores, pcx=1, pcy=2, conf_limit=0.99)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.6, label='Samples')
ax.plot(ellipse_95['x'], ellipse_95['y'], 'r-', linewidth=2, label='95% Confidence')
ax.plot(ellipse_99['x'], ellipse_99['y'], 'orange', linewidth=2, label='99% Confidence')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title("Hotelling's T² Ellipse from PCA Scores")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.show()
```

### Example 2: Confidence Ellipse from Raw Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyEllipse import confidence_ellipse

# Create sample dataset
np.random.seed(42)
df = pd.DataFrame({
    'SiO2': np.random.randn(150) + 70,
    'Na2O': np.random.randn(150) + 15,
    'type': np.repeat(['Glass_A', 'Glass_B', 'Glass_C'], 50)
})

# Add some structure to groups
df.loc[df['type'] == 'Glass_A', 'SiO2'] += 2
df.loc[df['type'] == 'Glass_B', 'Na2O'] += 2
df.loc[df['type'] == 'Glass_C', 'SiO2'] -= 1

# Compute confidence ellipses for each group
hotelling_coordinatess = confidence_ellipse(
    df, 
    x='SiO2', 
    y='Na2O', 
    group_by='type',
    conf_level=0.95,
    distribution='hotelling'
)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

colors = {'Glass_A': 'red', 'Glass_B': 'blue', 'Glass_C': 'green'}

# Plot data points
for glass_type, color in colors.items():
    subset = df[df['type'] == glass_type]
    ax.scatter(subset['SiO2'], subset['Na2O'], 
               c=color, alpha=0.5, s=50, label=f'{glass_type} data')

# Plot ellipses
for glass_type, color in colors.items():
    ellipse_subset = hotelling_coordinatess[hotelling_coordinatess['type'] == glass_type]
    ax.plot(ellipse_subset['x'], ellipse_subset['y'], 
            c=color, linewidth=2.5, label=f'{glass_type} 95% CI')

ax.set_xlabel('SiO2 (%)', fontsize=12)
ax.set_ylabel('Na2O (%)', fontsize=12)
ax.set_title('Confidence Ellipses by Glass Type', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.show()
```

### Example 3: 3D Ellipsoid Visualization

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pyEllipse import hotelling_coordinates

# Generate sample data
np.random.seed(42)
data = np.random.randn(100, 10)

# Perform PCA
pca = PCA(n_components=5)
pca_scores = pca.fit_transform(data)

# Generate 3D ellipsoid coordinates (fewer points for 3D)
ellipsoid = hotelling_coordinates(
    pca_scores, 
    pcx=1, 
    pcy=2, 
    pcz=3, 
    conf_limit=0.95,
    pts=40
)

# Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
ax.scatter(pca_scores[:, 0], pca_scores[:, 1], pca_scores[:, 2],
           alpha=0.6, s=30, label='Samples')

# Plot ellipsoid surface
grid_size = 40
x_surf = np.array(ellipsoid['x']).reshape(grid_size, grid_size)
y_surf = np.array(ellipsoid['y']).reshape(grid_size, grid_size)
z_surf = np.array(ellipsoid['z']).reshape(grid_size, grid_size)

ax.plot_surface(x_surf, y_surf, z_surf,
                alpha=0.3, color='red', edgecolor='none')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Hotelling Ellipsoid (95% Confidence)')
plt.tight_layout()
plt.show()
```

### Example 4: Outlier Detection with T²

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pyEllipse import hotelling_parameters, hotelling_coordinates

# Generate data with outliers
np.random.seed(42)
normal_data = np.random.randn(95, 10)
outliers = np.random.randn(5, 10) * 3 + 5
data = np.vstack([normal_data, outliers])

# Perform PCA
pca = PCA()
pca_scores = pca.fit_transform(data)

# Calculate T² statistics
results = hotelling_parameters(pca_scores, k=2)
t_squared = results['Tsquare']['value']
cutoff_95 = results['cutoff_95pct']

# Identify outliers
outliers_mask = t_squared > cutoff_95

# Generate ellipse
ellipse = hotelling_coordinates(pca_scores, pcx=1, pcy=2, conf_limit=0.95)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Score plot
ax1.scatter(pca_scores[~outliers_mask, 0], pca_scores[~outliers_mask, 1],
           c='blue', alpha=0.6, s=50, label='Normal')
ax1.scatter(pca_scores[outliers_mask, 0], pca_scores[outliers_mask, 1],
           c='red', alpha=0.8, s=100, marker='X', label='Outliers')
ax1.plot(ellipse['x'], ellipse['y'], 'g-', linewidth=2, label='95% Limit')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('PCA Score Plot with Outliers')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# T² plot
ax2.bar(range(len(t_squared)), t_squared, 
        color=['red' if x else 'blue' for x in outliers_mask],
        alpha=0.6)
ax2.axhline(y=cutoff_95, color='green', linestyle='--', 
           linewidth=2, label='95% Cutoff')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel("Hotelling's T²")
ax2.set_title('T² Values for Outlier Detection')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Number of outliers detected: {outliers_mask.sum()}")
print(f"Outlier indices: {np.where(outliers_mask)[0]}")
```

### Example 5: Comparing Normal vs Hotelling Distributions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyEllipse import confidence_ellipse

# Create small sample dataset (where distribution choice matters)
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(30),
    'y': np.random.randn(30)
})

# Compute ellipses with different distributions
ellipse_normal = confidence_ellipse(df, x='x', y='y', 
                                    conf_level=0.95, distribution='normal')
ellipse_hotelling = confidence_ellipse(df, x='x', y='y', 
                                       conf_level=0.95, distribution='hotelling')

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df['x'], df['y'], alpha=0.6, s=50, label='Data (n=30)')
ax.plot(ellipse_normal['x'], ellipse_normal['y'], 
        'b-', linewidth=2, label='Normal (χ²)')
ax.plot(ellipse_hotelling['x'], ellipse_hotelling['y'], 
        'r-', linewidth=2, label="Hotelling (T²)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Comparison: Normal vs Hotelling Distribution (Small Sample)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.show()
```

## Key Differences Between Functions

| Feature | `hotelling_parameters` | `hotelling_coordinates` | `confidence_ellipse` |
|---------|----------------|-----------------|---------------------|
| **Input** | Component scores | Component scores | Raw data |
| **Purpose** | T² statistics | Plot coordinates | Plot coordinates |
| **Grouping** | -- | -- | Yes |
| **Robust** | -- | -- | Yes |
| **2D/3D** | 2D only for ellipse params | Both | Both |
| **Distribution** | Hotelling only | Hotelling only | Normal or Hotelling |
| **Use Case** | Outlier detection, QC | Visualizing PCA | Exploratory data analysis |

## When to Use Each Function

### Use `hotelling_parameters` when:
- You need T² statistics for outlier detection
- You want confidence cutoff values
- You're performing quality control or process monitoring
- You need ellipse parameters (semi-axes lengths)

### Use `hotelling_coordinates` when:
- You have PCA/PLS component scores
- You want to visualize confidence regions on score plots
- You need precise control over which components to plot
- You're creating publication-quality figures from multivariate models

### Use `confidence_ellipse` when:
- You're working with raw data (not scores)
- You need to compare multiple groups
- You want robust estimation for outlier-resistant analysis
- You need flexibility in distribution choice (normal vs Hotelling)

## Dependencies

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Statistical functions
- `scikit-learn` - Robust covariance estimation (optional, for `robust=True`)
- `matplotlib` - Plotting (for examples)

## Statistical Background

### Hotelling's T² Distribution

Hotelling's T² statistic is the multivariate analog of the univariate Student's t-statistic. For sample size `n` and `p` dimensions:

```
T² = ((n - p) / (p(n - 1))) × MD²
```

where MD² is the squared Mahalanobis distance. The T² distribution accounts for uncertainty in estimating both the mean vector and covariance matrix from sample data, making it more appropriate than the chi-square distribution for small to moderate sample sizes.

### Distribution Choice

- **Normal (χ²)**: Assumes known population parameters. Appropriate for very large samples (n > 100).
- **Hotelling (T²/F)**: Accounts for parameter estimation uncertainty. Better for small samples (n < 100).

As sample size increases, the two distributions converge.

## References

1. Hotelling, H. (1931). The generalization of Student's ratio. *Annals of Mathematical Statistics*, 2(3), 360-378.

2. Brereton, R. G. (2016). Hotelling's T-squared distribution, its relationship to the F distribution and its use in multivariate space. *Journal of Chemometrics*, 30(1), 18-21.

3. Raymaekers, J., & Rousseeuw, P. J. (2019). Fast robust correlation for high dimensional data. *Technometrics*, 63(2), 184-198.

4. Jackson, J. E. (1991). *A User's Guide to Principal Components*. Wiley.

## License

MIT License

## Author

Christian L. Goueguel

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.
