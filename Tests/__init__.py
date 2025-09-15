"""
pyEllipse: Statistical confidence ellipses and Hotelling's T-squared ellipses.

This package provides tools for creating and analyzing confidence ellipses,
including Hotelling's T-squared ellipses for multivariate data analysis.
"""

# __version__ = "0.1.0"
# __author__ = “Christian L. Goueguel”
# __email__ = “christian.goueguel@gmail.com“

# Import main functions for easy access
# from .core.hotelling import hotelling_ellipse, hotelling_test
# from .core.confidence import confidence_ellipse, bivariate_confidence
# from .plotting.visualizations import plot_ellipse, plot_confidence_region

__all__ = [
    "hotelling_ellipse",
    "hotelling_test", 
    "confidence_ellipse",
    "bivariate_confidence",
    "plot_ellipse",
    "plot_confidence_region",
]