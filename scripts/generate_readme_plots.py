"""
Generate plots for GitHub README

Run this script once to generate all example plots as PNG files.
Then commit these images to your repository and reference them in README.md

Note: Type checking is disabled for this script due to matplotlib 3D plotting stubs.
"""
# type: ignore

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pathlib import Path



# Create directory for images
img_dir = Path("images")
img_dir.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')  # Modern style


def plot1_basic_hotelling_ellipse():
    """Example 1: Basic Hotelling's T² Ellipse"""
    np.random.seed(42)
    data = np.random.randn(100, 10)
    
    pca = PCA()
    pca_scores = pca.fit_transform(data)
    
    # Generate ellipse coordinates (simplified for demo)
    theta = np.linspace(0, 2 * np.pi, 361)
    mean_x, mean_y = np.mean(pca_scores[:, 0]), np.mean(pca_scores[:, 1])
    std_x, std_y = np.std(pca_scores[:, 0]), np.std(pca_scores[:, 1])
    
    # 95% and 99% ellipses
    ellipse_95_x = mean_x + 2.45 * std_x * np.cos(theta)
    ellipse_95_y = mean_y + 2.45 * std_y * np.sin(theta)
    ellipse_99_x = mean_x + 3.03 * std_x * np.cos(theta)
    ellipse_99_y = mean_y + 3.03 * std_y * np.sin(theta)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.6, s=50, 
               color='steelblue', edgecolors='navy', linewidth=0.5, label='Samples')
    ax.plot(ellipse_95_x, ellipse_95_y, 'r-', linewidth=2.5, label='95% Confidence')
    ax.plot(ellipse_99_x, ellipse_99_y, 'orange', linewidth=2.5, 
            linestyle='--', label='99% Confidence')
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title("Hotelling's T² Ellipse from PCA Scores", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(img_dir / 'example1_hotelling_ellipse.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: example1_hotelling_ellipse.png")


def plot2_grouped_ellipses():
    """Example 2: Grouped Confidence Ellipses"""
    np.random.seed(42)
    df = pd.DataFrame({
        'SiO2': np.random.randn(150) + 70,
        'Na2O': np.random.randn(150) + 15,
        'type': np.repeat(['Glass_A', 'Glass_B', 'Glass_C'], 50)
    })
    
    # Add structure to groups
    df.loc[df['type'] == 'Glass_A', 'SiO2'] += 2
    df.loc[df['type'] == 'Glass_B', 'Na2O'] += 2
    df.loc[df['type'] == 'Glass_C', 'SiO2'] -= 1
    
    colors = {'Glass_A': '#E74C3C', 'Glass_B': '#3498DB', 'Glass_C': '#2ECC71'}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data and ellipses for each group
    for glass_type, color in colors.items():
        subset = df[df['type'] == glass_type]
        
        # Plot points
        ax.scatter(subset['SiO2'], subset['Na2O'], 
                   c=color, alpha=0.5, s=60, edgecolors='black', 
                   linewidth=0.5, label=f'{glass_type} samples')
        
        # Generate simple ellipse
        mean_x = subset['SiO2'].mean()
        mean_y = subset['Na2O'].mean()
        std_x = subset['SiO2'].std()
        std_y = subset['Na2O'].std()
        
        theta = np.linspace(0, 2 * np.pi, 361)
        ellipse_x = mean_x + 2.45 * std_x * np.cos(theta)
        ellipse_y = mean_y + 2.45 * std_y * np.sin(theta)
        
        ax.plot(ellipse_x, ellipse_y, c=color, linewidth=3, 
                label=f'{glass_type} 95% CI')
    
    ax.set_xlabel('SiO₂ (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Na₂O (%)', fontsize=13, fontweight='bold')
    ax.set_title('Confidence Ellipses by Glass Type', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=10, frameon=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(img_dir / 'example2_grouped_ellipses.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: example2_grouped_ellipses.png")


# def plot3_3d_ellipsoid():
#     """Example 3: 3D Ellipsoid"""
#     np.random.seed(42)
#     data = np.random.randn(100, 10)
    
#     pca = PCA(n_components=5)
#     pca_scores = pca.fit_transform(data)
    
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # For 3D scatter plot, pass z-coordinates as third positional argument
#     # Extract coordinates to avoid type checker issues
#     xs = pca_scores[:, 0]
#     ys = pca_scores[:, 1]
#     zs = pca_scores[:, 2]
    
#     ax.scatter(xs, ys, zs,
#                alpha=0.6, s=40, c='steelblue', edgecolors='navy', 
#                linewidth=0.5, label='Samples')
    
#     # Generate simple ellipsoid surface
#     u = np.linspace(0, 2 * np.pi, 40)
#     v = np.linspace(0, np.pi, 40)
#     x_sphere = np.outer(np.cos(u), np.sin(v))
#     y_sphere = np.outer(np.sin(u), np.sin(v))
#     z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
#     # Scale by standard deviations
#     std_x = np.std(pca_scores[:, 0]) * 2.45
#     std_y = np.std(pca_scores[:, 1]) * 2.45
#     std_z = np.std(pca_scores[:, 2]) * 2.45
    
#     x_ellipsoid = x_sphere * std_x + np.mean(pca_scores[:, 0])
#     y_ellipsoid = y_sphere * std_y + np.mean(pca_scores[:, 1])
#     z_ellipsoid = z_sphere * std_z + np.mean(pca_scores[:, 2])
    
#     ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid,
#                     alpha=0.3, color='red', edgecolor='none')
    
#     ax.set_xlabel('PC1', fontsize=12, fontweight='bold', labelpad=10)
#     ax.set_ylabel('PC2', fontsize=12, fontweight='bold', labelpad=10)
#     ax.set_zlabel('PC3', fontsize=12, fontweight='bold', labelpad=10)
#     ax.set_title('3D Hotelling Ellipsoid (95% Confidence)', 
#                  fontsize=14, fontweight='bold', pad=20)
#     ax.view_init(elev=20, azim=45)
#     plt.tight_layout()
#     plt.savefig(img_dir / 'example3_3d_ellipsoid.png', dpi=300, bbox_inches='tight')
#     plt.close()
#     print("✓ Generated: example3_3d_ellipsoid.png")


def plot4_outlier_detection():
    """Example 4: Outlier Detection"""
    np.random.seed(42)
    normal_data = np.random.randn(95, 10)
    outliers = np.random.randn(5, 10) * 3 + 5
    data = np.vstack([normal_data, outliers])
    
    pca = PCA()
    pca_scores = pca.fit_transform(data)
    
    # Simple T² calculation
    mean_vec = np.mean(pca_scores[:, :2], axis=0)
    cov_mat = np.cov(pca_scores[:, :2].T)
    inv_cov = np.linalg.inv(cov_mat)
    
    t_squared = []
    for i in range(len(pca_scores)):
        diff = pca_scores[i, :2] - mean_vec
        t2 = diff @ inv_cov @ diff.T
        t_squared.append(t2)
    t_squared = np.array(t_squared)
    
    cutoff_95 = np.percentile(t_squared, 95)
    outliers_mask = t_squared > cutoff_95
    
    # Generate ellipse
    theta = np.linspace(0, 2 * np.pi, 361)
    mean_x, mean_y = mean_vec
    std_x = np.std(pca_scores[:, 0])
    std_y = np.std(pca_scores[:, 1])
    ellipse_x = mean_x + 2.45 * std_x * np.cos(theta)
    ellipse_y = mean_y + 2.45 * std_y * np.sin(theta)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Score plot
    ax1.scatter(pca_scores[~outliers_mask, 0], pca_scores[~outliers_mask, 1],
               c='steelblue', alpha=0.6, s=50, edgecolors='navy', 
               linewidth=0.5, label='Normal')
    ax1.scatter(pca_scores[outliers_mask, 0], pca_scores[outliers_mask, 1],
               c='red', alpha=0.9, s=150, marker='X', edgecolors='darkred',
               linewidth=1.5, label='Outliers')
    ax1.plot(ellipse_x, ellipse_y, 'g-', linewidth=2.5, label='95% Limit')
    ax1.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax1.set_title('PCA Score Plot with Outliers', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # T² plot
    colors_bar = ['red' if x else 'steelblue' for x in outliers_mask]
    ax2.bar(range(len(t_squared)), t_squared, color=colors_bar, alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax2.axhline(y=cutoff_95, color='green', linestyle='--', 
               linewidth=2.5, label='95% Cutoff')
    ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel("Hotelling's T²", fontsize=12, fontweight='bold')
    ax2.set_title('T² Values for Outlier Detection', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(img_dir / 'example4_outlier_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: example4_outlier_detection.png")


def plot5_distribution_comparison():
    """Example 5: Normal vs Hotelling Comparison"""
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(30),
        'y': np.random.randn(30)
    })
    
    theta = np.linspace(0, 2 * np.pi, 361)
    mean_x, mean_y = df['x'].mean(), df['y'].mean()
    std_x, std_y = df['x'].std(), df['y'].std()
    
    # Normal (chi-square based)
    normal_x = mean_x + 2.45 * std_x * np.cos(theta)
    normal_y = mean_y + 2.45 * std_y * np.sin(theta)
    
    # Hotelling (slightly larger for small n)
    hotelling_x = mean_x + 2.65 * std_x * np.cos(theta)
    hotelling_y = mean_y + 2.65 * std_y * np.sin(theta)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df['x'], df['y'], alpha=0.7, s=60, color='steelblue',
               edgecolors='navy', linewidth=0.8, label='Data (n=30)', zorder=3)
    ax.plot(normal_x, normal_y, 'b-', linewidth=2.5, label='Normal (χ²)')
    ax.plot(hotelling_x, hotelling_y, 'r-', linewidth=2.5, 
            linestyle='--', label="Hotelling (T²)")
    
    ax.fill_between(normal_x, normal_y, alpha=0.1, color='blue')
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_title('Distribution Comparison: Normal vs Hotelling (Small Sample)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=11, frameon=True, shadow=True, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add annotation
    ax.annotate('Hotelling gives\nwider bounds for\nsmall samples', 
                xy=(1.5, 1.5), xytext=(2.5, 2.5),
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                              color='black', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(img_dir / 'example5_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: example5_distribution_comparison.png")


def main():
    """Generate all plots"""
    print("Generating README plots...")
    print("-" * 50)
    
    plot1_basic_hotelling_ellipse()
    plot2_grouped_ellipses()
    # plot3_3d_ellipsoid()
    plot4_outlier_detection()
    plot5_distribution_comparison()
    
    print("-" * 50)
    print(f"✓ All plots saved to '{img_dir}/' directory")
    print("\nNext steps:")
    print("1. Commit the images/ directory to your repository")
    print("2. Update README.md to reference the images")
    print("\nExample markdown syntax:")
    print("![Hotelling Ellipse](images/example1_hotelling_ellipse.png)")


if __name__ == "__main__":

    main()

