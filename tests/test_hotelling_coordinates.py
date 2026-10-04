"""
Tests for `hotelling_coordinates`, ported from HotellingEllipse 1.3.0
(tests/testthat/test-ellipseCoord.R).
"""
import re

import numpy as np
import pandas as pd
import pytest

from pyEllipse import hotelling_coordinates, hotelling_parameters

from ._helpers import beta_limit, f_limit, mahalanobis_sq, pca_scores


@pytest.fixture
def scores():
    """Uncorrelated scores, as PCA scores are: 100 observations, 3 components."""
    return pca_scores(np.random.default_rng(123).standard_normal((100, 3)))


@pytest.fixture
def correlated():
    """Three correlated components."""
    z = np.random.default_rng(123).standard_normal((200, 3))
    return z @ np.array([[1, 0, 0], [0.5, 1, 0], [0.2, 0.4, 1]])


class TestOutput:
    def test_2d_structure(self, scores):
        res = hotelling_coordinates(scores)
        assert isinstance(res, pd.DataFrame)
        assert list(res.columns) == ["x", "y"]
        assert len(res) == 200

    def test_3d_structure(self, scores):
        res = hotelling_coordinates(scores, pcz=3, pts=20)
        assert list(res.columns) == ["x", "y", "z"]
        assert len(res) == 20 * 20

    def test_dataframe_and_array_agree(self, correlated):
        df = pd.DataFrame(correlated, columns=["PC1", "PC2", "PC3"])
        pd.testing.assert_frame_equal(hotelling_coordinates(df), hotelling_coordinates(correlated))

    def test_numpy_integer_arguments(self, correlated):
        pd.testing.assert_frame_equal(
            hotelling_coordinates(correlated, pcx=np.int64(1), pcy=np.int32(3), pts=np.int64(50)),
            hotelling_coordinates(correlated, pcx=1, pcy=3, pts=50),
        )


class TestGeometry:
    def test_ellipse_points_lie_on_the_tsquared_limit(self, correlated):
        xy = hotelling_coordinates(correlated, pcx=1, pcy=3, conf_limit=0.9, method="beta")
        md = mahalanobis_sq(correlated[:, [0, 2]], xy[["x", "y"]].to_numpy())
        np.testing.assert_allclose(md, beta_limit(200, 0.9))

    def test_ellipsoid_points_lie_on_the_tsquared_limit(self, correlated):
        xyz = hotelling_coordinates(correlated, pcz=3, pts=20)
        md = mahalanobis_sq(correlated, xyz[["x", "y", "z"]].to_numpy())
        np.testing.assert_allclose(md, f_limit(200, 0.95, k=3))

    def test_uncorrelated_scores_give_the_axis_aligned_ellipse(self, scores):
        # Same coordinates as pyEllipse <= 0.1.5
        xy = hotelling_coordinates(scores, pcx=2, pcy=3, pts=50)
        lim = f_limit(100, 0.95)
        theta = np.linspace(0, 2 * np.pi, 50)
        x, y = scores[:, 1], scores[:, 2]
        np.testing.assert_allclose(xy["x"], np.sqrt(lim * np.var(x, ddof=1)) * np.cos(theta) + x.mean(), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(xy["y"], np.sqrt(lim * np.var(y, ddof=1)) * np.sin(theta) + y.mean(), rtol=1e-8, atol=1e-12)

    def test_uncorrelated_scores_give_the_axis_aligned_ellipsoid(self, scores):
        xyz = hotelling_coordinates(scores, pcz=3, pts=10)
        lim = f_limit(100, 0.95, k=3)
        theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 10), np.linspace(0, np.pi, 10))
        theta, phi = theta.flatten(), phi.flatten()
        sd = np.sqrt(lim * np.var(scores, axis=0, ddof=1))
        mean = scores.mean(axis=0)
        np.testing.assert_allclose(xyz["x"], sd[0] * np.cos(theta) * np.sin(phi) + mean[0], rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(xyz["y"], sd[1] * np.sin(theta) * np.sin(phi) + mean[1], rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(xyz["z"], sd[2] * np.cos(phi) + mean[2], rtol=1e-8, atol=1e-12)

    def test_matches_the_hotelling_parameters_ellipse(self, correlated):
        params = hotelling_parameters(correlated, pcx=1, pcy=2, conf_limit=0.99, method="beta")
        xy = hotelling_coordinates(correlated, pcx=1, pcy=2, conf_limit=0.99, pts=20001, method="beta")
        # Same T-squared limit
        md = mahalanobis_sq(correlated[:, :2], xy[["x", "y"]].to_numpy())
        np.testing.assert_allclose(md, params["cutoff_99pct"])
        # Same semi-axes and rotation
        angle = params["Ellipse"]["angle"][0]
        xc = xy[["x", "y"]].to_numpy() - correlated[:, :2].mean(axis=0)
        u = np.cos(angle) * xc[:, 0] + np.sin(angle) * xc[:, 1]
        v = -np.sin(angle) * xc[:, 0] + np.cos(angle) * xc[:, 1]
        assert np.max(np.abs(u)) == pytest.approx(params["Ellipse"]["a_99pct"][0], rel=1e-6)
        assert np.max(np.abs(v)) == pytest.approx(params["Ellipse"]["b_99pct"][0], rel=1e-6)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"conf_limit": 1.5}, "Confidence level should be a numeric value between 0 and 1."),
            ({"conf_limit": 0}, "Confidence level should be a numeric value between 0 and 1."),
            ({"conf_limit": np.nan}, "Confidence level should be a numeric value between 0 and 1."),
            ({"pcx": 0}, "'pcx' must be an integer between 1 and the number of components in the data (3)."),
            ({"pcx": 200}, "'pcx' must be an integer between 1"),
            ({"pcy": 0}, "'pcy' must be an integer between 1"),
            ({"pcy": 200}, "'pcy' must be an integer between 1"),
            ({"pcx": 1, "pcy": 1}, "'pcx' and 'pcy' must be different integers."),
            ({"pts": 0}, "'pts' should be a positive integer."),
            ({"pts": -10}, "'pts' should be a positive integer."),
            ({"pts": 2.5}, "'pts' should be a positive integer."),
            ({"pcz": 0}, "'pcz' must be an integer between 1"),
            ({"pcz": 200}, "'pcz' must be an integer between 1"),
            ({"pcx": 1, "pcy": 2, "pcz": 1}, "'pcx', 'pcy', and 'pcz' must be different integers."),
            ({"pcx": 1, "pcy": 2, "pcz": 2}, "'pcx', 'pcy', and 'pcz' must be different integers."),
            ({"method": "chisq"}, "'method' must be either 'f' or 'beta'."),
        ],
    )
    def test_invalid_arguments(self, scores, kwargs, message):
        with pytest.raises(ValueError, match=re.escape(message)):
            hotelling_coordinates(scores, **kwargs)

    def test_missing_input(self):
        with pytest.raises(ValueError, match="Missing input data."):
            hotelling_coordinates(None)

    def test_invalid_input_type(self):
        with pytest.raises(TypeError, match="numpy array or pandas DataFrame"):
            hotelling_coordinates(list(range(100)))

    def test_one_dimensional_input(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            hotelling_coordinates(np.arange(100.0))

    def test_too_few_observations(self, scores):
        with pytest.raises(ValueError, match="At least 4 observations are needed to use 2 components"):
            hotelling_coordinates(scores[:3])
        with pytest.raises(ValueError, match="At least 5 observations are needed to use 3 components"):
            hotelling_coordinates(scores[:4], pcz=3)
