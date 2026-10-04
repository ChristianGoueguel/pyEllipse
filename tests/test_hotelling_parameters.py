"""
Tests for `hotelling_parameters`, ported from HotellingEllipse 1.3.0
(tests/testthat/test-ellipseParam.R).
"""
import re

import numpy as np
import pandas as pd
import pytest

from pyEllipse import hotelling_parameters

from ._helpers import beta_limit, f_limit, mahalanobis_sq, pca_scores


@pytest.fixture
def scores():
    """Uncorrelated scores, as PCA scores are: 100 observations, 4 components."""
    return pca_scores(np.random.default_rng(123).standard_normal((100, 4)))


@pytest.fixture
def correlated():
    """Two correlated components (r = 0.8), e.g. new samples projected onto a model."""
    z = np.random.default_rng(123).standard_normal((200, 2))
    return np.column_stack([z[:, 0], 0.8 * z[:, 0] + 0.6 * z[:, 1]])


def outside_ellipse(x, ellipse, label):
    """True for the points of the two-column x lying outside the ellipse `ellipse`."""
    xc = x - x.mean(axis=0)
    angle = ellipse["angle"][0]
    u = np.cos(angle) * xc[:, 0] + np.sin(angle) * xc[:, 1]
    v = -np.sin(angle) * xc[:, 0] + np.cos(angle) * xc[:, 1]
    return (u / ellipse[f"a_{label}"][0]) ** 2 + (v / ellipse[f"b_{label}"][0]) ** 2 > 1


def assert_same_result(res1, res2):
    assert list(res1) == list(res2)
    for key in res1:
        if isinstance(res1[key], pd.DataFrame):
            pd.testing.assert_frame_equal(res1[key], res2[key])
        else:
            assert res1[key] == res2[key]


class TestOutput:
    def test_structure(self, scores):
        res = hotelling_parameters(scores)
        assert list(res) == ["Tsquared", "cutoff_99pct", "cutoff_95pct", "nb_comp", "Ellipse"]
        assert list(res["Tsquared"].columns) == ["value"]
        assert len(res["Tsquared"]) == len(scores)
        assert list(res["Ellipse"].columns) == ["a_99pct", "b_99pct", "a_95pct", "b_95pct", "angle"]
        assert type(res["nb_comp"]) is int and res["nb_comp"] == 2
        assert type(res["cutoff_95pct"]) is float

    def test_no_ellipse_when_k_is_above_2(self, scores):
        res = hotelling_parameters(scores, k=3)
        assert list(res) == ["Tsquared", "cutoff_99pct", "cutoff_95pct", "nb_comp"]
        assert res["nb_comp"] == 3

    def test_dataframe_and_array_agree(self, scores):
        df = pd.DataFrame(scores, columns=["PC1", "PC2", "PC3", "PC4"])
        assert_same_result(hotelling_parameters(df), hotelling_parameters(scores))

    def test_numpy_integer_arguments(self, scores):
        res = hotelling_parameters(scores, k=np.int64(2), pcx=np.int32(1), pcy=np.int64(3))
        assert_same_result(res, hotelling_parameters(scores, pcx=1, pcy=3))


class TestTsquaredScale:
    def test_tsquared_is_the_squared_mahalanobis_distance(self):
        x = np.random.default_rng(123).standard_normal((50, 3))
        res = hotelling_parameters(x)
        np.testing.assert_allclose(res["Tsquared"]["value"], mahalanobis_sq(x[:, :2]))
        assert res["cutoff_95pct"] == pytest.approx(f_limit(50, 0.95))
        assert res["cutoff_99pct"] == pytest.approx(f_limit(50, 0.99))

    def test_k_components(self, scores):
        res = hotelling_parameters(scores, k=3)
        np.testing.assert_allclose(res["Tsquared"]["value"], mahalanobis_sq(scores[:, :3]))
        assert res["cutoff_95pct"] == pytest.approx(f_limit(100, 0.95, k=3))

    def test_flagged_fraction_matches_the_confidence_level(self):
        # pyEllipse <= 0.1.5 returned (n - k) / (k(n - 1)) * T^2 while the cutoffs are on
        # the T^2 scale, so about 0.04% of the points exceeded the 95% cutoff instead of 5%.
        # The beta limit is exact for the observations the mean and covariance come from.
        rng = np.random.default_rng(1)
        flagged = []
        for _ in range(200):
            res = hotelling_parameters(rng.standard_normal((50, 3)), method="beta")
            flagged.append(np.mean(res["Tsquared"]["value"] > res["cutoff_95pct"]))
        assert np.mean(flagged) == pytest.approx(0.05, abs=0.01)

    def test_uses_pcx_and_pcy_when_k_is_2(self, scores):
        res = hotelling_parameters(scores, pcx=2, pcy=4)
        x2 = scores[:, [1, 3]]
        np.testing.assert_allclose(res["Tsquared"]["value"], mahalanobis_sq(x2))
        # Points outside the 95% ellipse are exactly those above the 95% cutoff
        np.testing.assert_array_equal(
            outside_ellipse(x2, res["Ellipse"], "95pct"),
            res["Tsquared"]["value"] > res["cutoff_95pct"],
        )


class TestMethod:
    def test_method_selects_the_limit(self, scores):
        beta = hotelling_parameters(scores, method="beta")
        f = hotelling_parameters(scores)
        assert beta["cutoff_99pct"] == pytest.approx(beta_limit(100, 0.99))
        assert f["cutoff_99pct"] == pytest.approx(f_limit(100, 0.99))
        pd.testing.assert_frame_equal(beta["Tsquared"], f["Tsquared"])

    def test_beta_limit_never_exceeds_the_largest_attainable_tsquared(self, scores):
        small = scores[:10]
        max_tsq = (10 - 1) ** 2 / 10
        assert hotelling_parameters(small, method="beta")["cutoff_99pct"] < max_tsq
        assert hotelling_parameters(small)["cutoff_99pct"] > max_tsq


class TestEllipse:
    def test_correlated_scores_rotate_the_ellipse(self, correlated):
        res = hotelling_parameters(correlated)
        assert abs(res["Ellipse"]["angle"][0]) > 0.1
        for label in ("95pct", "99pct"):
            np.testing.assert_array_equal(
                outside_ellipse(correlated, res["Ellipse"], label),
                res["Tsquared"]["value"] > res[f"cutoff_{label}"],
            )

    def test_uncorrelated_scores_give_the_axis_aligned_ellipse(self, correlated):
        # Same semi-axes as pyEllipse <= 0.1.5
        pca = pca_scores(correlated)
        res = hotelling_parameters(pca)
        ellipse = res["Ellipse"]
        assert ellipse["angle"][0] == 0
        assert ellipse["a_95pct"][0] == pytest.approx(np.sqrt(res["cutoff_95pct"] * np.var(pca[:, 0], ddof=1)))
        assert ellipse["b_95pct"][0] == pytest.approx(np.sqrt(res["cutoff_95pct"] * np.var(pca[:, 1], ddof=1)))

    def test_equal_variance_uncorrelated_scores_give_angle_0(self):
        # e.g. SIMPLS scores, which are scaled to equal variance: the ellipse is a circle
        pca = pca_scores(np.random.default_rng(123).standard_normal((100, 2)))
        pca = pca / pca.std(axis=0, ddof=1)
        ellipse = hotelling_parameters(pca)["Ellipse"]
        assert ellipse["angle"][0] == 0
        assert ellipse["a_95pct"][0] == pytest.approx(ellipse["b_95pct"][0])

    def test_a_is_the_semi_axis_closest_to_pcx(self):
        z = np.random.default_rng(7).standard_normal((100, 2))
        x = np.column_stack([2 * z[:, 0], z[:, 0] + 0.5 * z[:, 1]])
        xy = hotelling_parameters(x, pcx=1, pcy=2)["Ellipse"]
        yx = hotelling_parameters(x, pcx=2, pcy=1)["Ellipse"]
        assert abs(xy["angle"][0]) <= np.pi / 4
        assert yx["a_95pct"][0] == pytest.approx(xy["b_95pct"][0])
        assert yx["b_95pct"][0] == pytest.approx(xy["a_95pct"][0])
        assert yx["angle"][0] == pytest.approx(-xy["angle"][0])


class TestConfLimit:
    def test_default_output_is_unchanged(self, scores):
        res = hotelling_parameters(scores)
        assert_same_result(res, hotelling_parameters(scores, conf_limit=(0.95, 0.99)))
        assert_same_result(res, hotelling_parameters(scores, conf_limit=[0.99, 0.95]))
        assert_same_result(res, hotelling_parameters(scores, conf_limit=np.array([0.95, 0.99])))

    def test_custom_levels_are_named_from_highest_to_lowest(self, scores):
        res = hotelling_parameters(scores, conf_limit=(0.975, 0.999))
        assert list(res) == ["Tsquared", "cutoff_99.9pct", "cutoff_97.5pct", "nb_comp", "Ellipse"]
        assert list(res["Ellipse"].columns) == ["a_99.9pct", "b_99.9pct", "a_97.5pct", "b_97.5pct", "angle"]
        assert res["cutoff_97.5pct"] == pytest.approx(f_limit(100, 0.975))
        assert res["Ellipse"]["a_99.9pct"][0] == pytest.approx(np.sqrt(f_limit(100, 0.999) * np.var(scores[:, 0], ddof=1)))
        assert res["Ellipse"]["b_97.5pct"][0] == pytest.approx(np.sqrt(f_limit(100, 0.975) * np.var(scores[:, 1], ddof=1)))

    def test_single_and_multiple_levels(self, scores):
        res = hotelling_parameters(scores, conf_limit=0.9)
        assert list(res) == ["Tsquared", "cutoff_90pct", "nb_comp", "Ellipse"]
        assert list(res["Ellipse"].columns) == ["a_90pct", "b_90pct", "angle"]

        res = hotelling_parameters(scores, k=3, conf_limit=(0.9, 0.95, 0.99))
        assert list(res) == ["Tsquared", "cutoff_99pct", "cutoff_95pct", "cutoff_90pct", "nb_comp"]
        assert res["cutoff_90pct"] == pytest.approx(f_limit(100, 0.9, k=3))

        res = hotelling_parameters(scores, threshold=0.9, conf_limit=0.975, method="beta")
        assert list(res) == ["Tsquared", "cutoff_97.5pct", "nb_comp"]

    @pytest.mark.parametrize(
        "conf_limit", [1, 0, (0.95, np.nan), [], "0.95", [[0.9, 0.95]], True, None]
    )
    def test_invalid_levels(self, scores, conf_limit):
        with pytest.raises(ValueError, match="'conf_limit' must be a number or a sequence"):
            hotelling_parameters(scores, conf_limit=conf_limit)

    def test_duplicated_levels(self, scores):
        with pytest.raises(ValueError, match="must not contain duplicated values"):
            hotelling_parameters(scores, conf_limit=(0.95, 0.95))


class TestComponentSelection:
    def test_threshold_1_selects_all_components(self, scores):
        assert hotelling_parameters(scores, threshold=1)["nb_comp"] == 4

    def test_threshold_removes_near_zero_variance_component(self, scores):
        scores[:, 1] *= 1e-4
        with pytest.warns(UserWarning, match="component 2 removed"):
            res = hotelling_parameters(scores, threshold=1)
        assert res["nb_comp"] == 3
        np.testing.assert_allclose(res["Tsquared"]["value"], mahalanobis_sq(scores[:, [0, 2, 3]]))

    def test_threshold_lower_than_first_component(self, scores):
        with pytest.warns(UserWarning, match="lower than the variance explained by the first component"):
            res = hotelling_parameters(scores, threshold=0.1)
        assert res["nb_comp"] == 2

    def test_fixed_k_removes_near_zero_variance_component(self, scores):
        scores[:, 1] *= 1e-4
        df = pd.DataFrame(scores, columns=["PC1", "PC2", "PC3", "PC4"])
        with pytest.warns(UserWarning, match="detected: PC2 removed"):
            res = hotelling_parameters(df, k=3)
        assert res["nb_comp"] == 3
        np.testing.assert_allclose(res["Tsquared"]["value"], mahalanobis_sq(scores[:, [0, 2, 3]]))

    def test_fixed_k_with_fewer_than_two_remaining_components(self, scores):
        x = scores[:, :3].copy()
        x[:, 1:] *= 1e-4
        with pytest.warns(UserWarning, match="removed"):
            with pytest.raises(ValueError, match="Fewer than two components remain"):
                hotelling_parameters(x, k=3)

    def test_threshold_with_fewer_than_two_remaining_components(self, scores):
        x = scores[:, :3].copy()
        x[:, 1:] *= 1e-3
        with pytest.warns(UserWarning, match="removed"):
            with pytest.raises(ValueError, match="Fewer than two components remain"):
                hotelling_parameters(x, threshold=1)

    @pytest.mark.parametrize("axis", ["pcx", "pcy"])
    def test_near_zero_variance_ellipse_component(self, scores, axis):
        scores[:, 2] *= 1e-4
        kwargs = {"pcx": 3, "pcy": 1} if axis == "pcx" else {"pcx": 1, "pcy": 3}
        with pytest.raises(ValueError, match=f"'{axis}' has a relative variance lower than 'rel_tol'"):
            hotelling_parameters(scores, **kwargs)

    def test_singular_covariance(self):
        # Integer data with integer means, so the duplicated column makes the
        # covariance matrix exactly singular
        x = np.array([[1, 2, 1], [3, 1, 3], [-2, 0, -2], [0, -3, 0], [-2, 0, -2]], dtype=float)
        with pytest.raises(RuntimeError, match="Error in T-squared calculation"):
            hotelling_parameters(x, k=3)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, error, message",
        [
            ({"rel_tol": -0.1}, ValueError, "'rel_tol' must be a non-negative numeric value."),
            ({"rel_tol": np.nan}, ValueError, "'rel_tol' must be a non-negative numeric value."),
            ({"abs_tol": -0.1}, ValueError, "'abs_tol' must be a non-negative numeric value."),
            ({"rel_tol": 0.001, "abs_tol": 0.01}, ValueError, "'abs_tol' must be less than or equal to 'rel_tol'."),
            ({"threshold": 1.5}, ValueError, "Threshold must be a numeric value between 0 and 1."),
            ({"threshold": 0}, ValueError, "Threshold must be a numeric value between 0 and 1."),
            ({"threshold": np.nan}, ValueError, "Threshold must be a numeric value between 0 and 1."),
            ({"k": 1}, ValueError, "'k' must be an integer between 2 and the number of components in the data (4)."),
            ({"k": 5}, ValueError, "'k' must be an integer between 2 and the number of components in the data (4)."),
            ({"k": 2.5}, ValueError, "'k' must be an integer"),
            ({"k": True}, ValueError, "'k' must be an integer"),
            ({"pcx": 0}, ValueError, "'pcx' must be an integer between 1 and the number of components in the data (4)."),
            ({"pcx": 200}, ValueError, "'pcx' must be an integer between 1"),
            ({"pcx": [1, 2]}, ValueError, "'pcx' must be an integer"),
            ({"pcy": 0}, ValueError, "'pcy' must be an integer between 1"),
            ({"pcy": 200}, ValueError, "'pcy' must be an integer between 1"),
            ({"pcx": 1, "pcy": 1}, ValueError, "'pcx' and 'pcy' must be different integers."),
            ({"method": "chisq"}, ValueError, "'method' must be either 'f' or 'beta'."),
        ],
    )
    def test_invalid_arguments(self, scores, kwargs, error, message):
        with pytest.raises(error, match=re.escape(message)):
            hotelling_parameters(scores, **kwargs)

    def test_missing_input(self):
        with pytest.raises(ValueError, match="Missing input data."):
            hotelling_parameters(None)

    def test_invalid_input_type(self):
        with pytest.raises(TypeError, match="numpy array or pandas DataFrame"):
            hotelling_parameters(list(range(100)))

    def test_one_dimensional_input(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            hotelling_parameters(np.arange(100.0))

    def test_too_few_observations(self, scores):
        with pytest.raises(ValueError, match=re.escape("At least 4 observations are needed to use 2 components (got 3).")):
            hotelling_parameters(scores[:3])
