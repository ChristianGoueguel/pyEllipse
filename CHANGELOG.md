# Changelog

## pyEllipse 0.2.1

### Bug fixes

- `confidence_ellipse()` with `robust=True` now gives regions with the nominal confidence level for normal data with scikit-learn < 1.8. These versions do not make the reweighted Minimum Covariance Determinant covariance matrix consistent at the normal distribution, so its variances were about 10% too small, and a 95% robust ellipse contained only about 93% of the distribution. pyEllipse now applies the consistency factor of scikit-learn ≥ 1.8 (Croux and Haesbroeck, 1999) with older versions, so robust results no longer depend on the version of scikit-learn.

- `confidence_ellipse()` now requires at least four observations (of each group) for an ellipsoid. With three, `distribution='hotelling'` failed with a `ZeroDivisionError`, and `distribution='normal'` returned a flat ellipsoid.

- `confidence_ellipse()` no longer returns NaN coordinates for collinear variables, whose covariance matrix is singular: the region is then flat (a line segment for two variables).

- With a categorical `group_by` column, `confidence_ellipse()` no longer fails with "At least 3 observations are required" when some categories have no observations, e.g. after filtering the data, and no longer issues a pandas `FutureWarning`.

- `conf_level=nan` in `confidence_ellipse()` now raises an error instead of returning NaN coordinates, and NumPy floating-point scalars such as `np.float32` are now accepted.

## pyEllipse 0.2.0

This release ports the fixes and features of the [HotellingEllipse](https://github.com/ChristianGoueguel/HotellingEllipse) 1.3.0 R package.

### Breaking changes

- `hotelling_parameters()` now returns `Tsquared['value']` as Hotelling's T-squared statistic (the squared Mahalanobis distance). Previously it returned the F-scaled statistic, (n − k) / (k(n − 1)) × T², while `cutoff_95pct` and `cutoff_99pct` were on the T-squared scale. Comparing `value` against the cutoffs therefore used a threshold that was too high by a factor of k(n − 1) / (n − k) (about 2 for n = 50, k = 2), so outliers were missed. `value` and the cutoffs are now on the same scale, and a point lies outside the ellipse exactly when `value` exceeds the corresponding cutoff.

- `hotelling_parameters()` returns an additional `angle` column in `Ellipse`.

- Messages about removed near-zero variance components and low thresholds are now issued as warnings (`UserWarning`) instead of being printed. They name the removed components by column name (DataFrame input) or 1-based component number, instead of 0-based indices.

### New features

- The ellipse (ellipsoid) is now rotated when the selected components are correlated, so that it always matches the T-squared statistic. Previously it was always axis-aligned. The scores of the samples a PCA or PLS model was fitted on are uncorrelated, so for them `angle` is 0 and the results are identical to previous versions. Rotation only applies to correlated scores, e.g. new samples projected onto a model, PLS Y-scores, rotated (varimax) or ICA components.

- New `method` argument in `hotelling_parameters()` and `hotelling_coordinates()` to choose the T-squared limit. The default, `method='f'`, is the limit used in previous versions, k(n − 1) / (n − k) × F(k, n − k), so cutoffs and ellipses are unchanged. `method='beta'` uses the exact distribution of T-squared for the observations used to estimate the mean and covariance, (n − 1)² / n × Beta(k/2, (n − k − 1)/2) (Tracy, Young and Mason, 1992), e.g. the scores of the samples a PCA or PLS model was built on. The F-based limit is more conservative for small n: with n = 10 and k = 2, the 99% F limit is 19.5, while no observation can have T-squared above (n − 1)² / n = 8.1.

- New `conf_limit` argument in `hotelling_parameters()` to set the confidence levels of the T-squared cutoffs and ellipse semi-axes. It accepts a single level or a sequence of any number of levels and defaults to `(0.95, 0.99)`, which gives the same output as before. Results are named after each level, from the highest to the lowest: `conf_limit=(0.975, 0.999)` returns `cutoff_99.9pct` and `cutoff_97.5pct`, and `Ellipse` columns `a_99.9pct`, `b_99.9pct`, `a_97.5pct` and `b_97.5pct`.

### Bug fixes

- When `k=2`, `hotelling_parameters()` now computes `Tsquared` on components `pcx` and `pcy`, the same components used for the ellipse. Previously it always used the first two components.

- With `threshold`, removing near-zero variance components could leave a single component, which failed with a cryptic `LinAlgError`. `threshold=1` could also fail due to floating-point rounding ("Threshold is too high"). Both cases are now handled. With a fixed `k`, an error is raised when fewer than two components remain after removing near-zero variance components.

- Non-integer or boolean values of `k`, `pcx`, `pcy`, `pcz` and `pts`, NaN values of `threshold`, `rel_tol`, `abs_tol` and `conf_limit`, and one-dimensional input now give informative errors. An error is also raised when there are too few observations for the number of components. NumPy integers are now accepted for `k`, `pcx`, `pcy`, `pcz` and `pts`.

- `nb_comp` is now returned as a Python `int`.

### Other changes

- Added a test suite (`tests/`), including parity tests against HotellingEllipse 1.3.0 (`tests/r_parity_reference.R`). The examples in the docstrings are run as tests too.

- New documentation website, <https://christiangoueguel.com/pyEllipse>, built with Quarto and quartodoc and published by GitHub Actions: the theory behind each function (with derivations, the exact distribution of T-squared in each case, and simulation checks), worked examples, and the API reference. Figures, tables and numbers are computed with pyEllipse when the site is built, and every page shows the version and commit it was built from.

- The docstrings of `hotelling_parameters()`, `hotelling_coordinates()` and `confidence_ellipse()` follow the numpydoc format, with the formulas and runnable examples.

- The README examples use the wine data of scikit-learn, and their figures are publication ready (journal column sizes, embedded fonts, readable in grayscale). `scripts/readme_figures.py` regenerates the figures from the README's own code, and `scripts/examples_notebook.py` regenerates `examples.ipynb` from the README examples.

- The pages of the former documentation (`pyEllipse.html` and `pyEllipse/<function>.html`) redirect to the new site.
