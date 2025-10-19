"""
**Statistical confidence ellipses and Hotelling's T-squared statistic**

The pyEllipse library provides robust computational tools for generating and analyzing 
confidence ellipses and ellipsoids in multivariate datasets. Built on rigorous statistical 
foundations, it supports both classical normal-based confidence regions and Hotelling's 
T-squared ellipses, delivering robust uncertainty quantification for small samples, outlier 
detection, process control, and high-dimensional data exploration.
"""

# Import main functions with their proper names
from .hotelling_parameters import hotelling_parameters
from .hotelling_coordinates import hotelling_coordinates
from .confidence_ellipse import confidence_ellipse

__all__ = [
    "hotelling_parameters",
    "hotelling_coordinates", 
    "confidence_ellipse",
]

# Library metadata
__title__ = "pyEllipse"
__description__ = "Statistical confidence ellipses and Hotelling's T-squared ellipses"
__url__ = "https://github.com/ChristianGoueguel/pyEllipse"
__license__ = "MIT"
__version__ = "0.1.1"
__author__ = "Christian L. Goueguel"
__maintainer__ = "Christian L. Goueguel"
__credits__ = ["Christian L. Goueguel"] 
__email__ = "christian.goueguel@gmail.com"