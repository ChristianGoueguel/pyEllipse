"""
Setup script for pyEllipse package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="pyEllipse",
    version="0.1.5",
    author="Christian L. Goueguel",
    author_email="christian.goueguel@gmail.com",
    description=(
        "Tools for creating and analyzing confidence ellipses, including "
        "Hotelling's T-squared ellipses for multivariate statistical "
        "analysis and data visualization."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChristianGoueguel/pyEllipse",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="statistics confidence-ellipse hotelling multivariate visualization",
    python_requires=">=3.9,<3.15",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "plotting": [
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
        "all": [
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/ChristianGoueguel/pyEllipse/issues",
        "Source": "https://github.com/ChristianGoueguel/pyEllipse",
        "Documentation": "https://christiangoueguel.github.io/pyEllipse",
    },
)