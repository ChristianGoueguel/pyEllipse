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
    version="0.1.0",
    author="Christian L. Goueguel",
    author_email="christian.goueguel@gmail.com",
    description="Hotelling's T² statistics and confidence ellipse analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyEllipse",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="hotelling t-squared multivariate statistics confidence ellipse pca chemometrics",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.5.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pyEllipse/issues",
        "Source": "https://github.com/yourusername/pyEllipse",
        "Documentation": "https://github.com/yourusername/pyEllipse#readme",
    },
)