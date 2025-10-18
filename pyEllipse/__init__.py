"""
pyEllipse: Statistical confidence ellipses and Hotelling's T-squared ellipses.

This package provides tools for creating and analyzing confidence ellipses,
including Hotelling's T-squared ellipses for multivariate data analysis.
"""

__version__ = "0.1.1"
__author__ = "Christian L. Goueguel"
__email__ = "christian.goueguel@gmail.com"

# Import main functions with their proper names
from .hotelling_parameters import hotelling_parameters
from .hotelling_coordinates import hotelling_coordinates
from .confidence_ellipse import confidence_ellipse

__all__ = [
    "hotelling_parameters",
    "hotelling_coordinates", 
    "confidence_ellipse",
]

# Package metadata
__title__ = "pyEllipse"
__description__ = "Statistical confidence ellipses and Hotelling's T-squared ellipses"
__url__ = "https://github.com/clgoueguel/pyEllipse"
__license__ = "MIT"