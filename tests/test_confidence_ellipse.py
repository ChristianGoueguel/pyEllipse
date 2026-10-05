"""
Tests for `confidence_ellipse`.

The points are checked against the mean, covariance matrix and chi-square or F quantile
computed independently of the package: a point on the boundary of the region has a
squared Mahalanobis distance from the mean equal to the quantile.
"""
import importlib
import re

import numpy as np
import pandas as pd
import pytest
import sklearn
from packaging.version import Version
from scipy import stats
from sklearn.covariance import MinCovDet

from pyEllipse import confidence_ellipse, hotelling_parameters

from ._helpers import f_limit, mahalanobis_sq

# The module, which the function of the same name hides in the package namespace
confidence_ellipse_module = importlib.import_module("pyEllipse.confidence_ellipse")

SKLEARN_BEFORE_1_8 = Version(sklearn.__version__) < Version("1.8")

MIXING = np.array([[2, 0, 0], [1.2, 1, 0], [-0.5, 0.7, 0.4]])


def quantile(distribution: str, level: float, n: int, p: int) -> float:
    """Chi-square or Hotelling's T-squared quantile of the region of p variables."""
    if distribution == "normal":
        return stats.chi2.ppf(level, p)
    return f_limit(n, level, k=p)


@pytest.fixture
def data():
    """Three correlated variables with unequal variances and nonzero means: 40 observations."""
    z = np.random.default_rng(123).standard_normal((40, 3))
    return pd.DataFrame(z @ MIXING.T + [3, -2, 1], columns=["u", "v", "w"])


@pytest.fixture
def grouped():
    """
    Two groups of different sizes, means and covariance matrices, with interleaved rows:
    each region must come from the observations, and the number, of its own group.
    """
    rng = np.random.default_rng(42)
    a = rng.standard_normal((12, 3)) @ np.array([[1, 0, 0], [0.8, 0.5, 0], [0, 0.3, 1]]).T
    b = rng.standard_normal((28, 3)) @ np.array([[0.5, 0, 0], [-1, 2, 0], [0.2, 0, 0.3]]).T + [5, 5, -5]
    df = pd.DataFrame(np.vstack([a, b]), columns=["u", "v", "w"])
    df["group"] = ["A"] * 12 + ["B"] * 28
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture
def contaminated():
    """200 observations of three correlated variables, the first 10 replaced by outliers."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((200, 3)) @ np.array([[1, 0, 0], [0.6, 0.8, 0], [0.2, -0.3, 0.9]]).T
    x[:10] = 6 + 0.1 * rng.standard_normal((10, 3))
    return pd.DataFrame(x, columns=["u", "v", "w"])


def variables(z):
    return ["u", "v"] if z is None else ["u", "v", "w"]


def reweighted_mcd(x):
    """
    Reweighted MCD estimates of x, consistent at the normal distribution: mean and covariance
    matrix of the observations within the 97.5% chi-square quantile of the raw MCD distances,
    the latter multiplied by the factor of Croux and Haesbroeck (1999).
    """
    p = x.shape[1]
    kept = x[MinCovDet(support_fraction=0.9, random_state=42).fit(x).support_]
    factor = 0.975 / stats.chi2.cdf(stats.chi2.ppf(0.975, p), p + 2)
    return kept.mean(axis=0), np.cov(kept, rowvar=False, bias=True) * factor


class TestOutput:
    def test_ellipse_structure(self, data):
        res = confidence_ellipse(data, x="u", y="v")
        assert isinstance(res, pd.DataFrame)
        assert list(res.columns) == ["x", "y"]
        assert len(res) == 361
        assert np.isfinite(res.to_numpy()).all()

    def test_ellipsoid_structure(self, data):
        res = confidence_ellipse(data, x="u", y="v", z="w")
        assert list(res.columns) == ["x", "y", "z"]
        assert len(res) == 50 * 50
        assert np.isfinite(res.to_numpy()).all()

    def test_ellipse_is_closed(self, data):
        res = confidence_ellipse(data, x="u", y="v")
        np.testing.assert_allclose(res.iloc[-1], res.iloc[0])

    def test_numpy_float64_conf_level(self, data):
        pd.testing.assert_frame_equal(
            confidence_ellipse(data, x="u", y="v", conf_level=np.float64(0.9)),
            confidence_ellipse(data, x="u", y="v", conf_level=0.9),
        )

    def test_numpy_float32_conf_level(self, data):
        level = np.float32(0.9)
        pd.testing.assert_frame_equal(
            confidence_ellipse(data, x="u", y="v", conf_level=level),
            confidence_ellipse(data, x="u", y="v", conf_level=float(level)),
        )


class TestEllipse:
    @pytest.mark.parametrize("distribution", ["normal", "hotelling"])
    @pytest.mark.parametrize("level", [0.5, 0.95, 0.999])
    def test_points_lie_on_the_quantile(self, data, distribution, level):
        res = confidence_ellipse(data, x="u", y="w", conf_level=level, distribution=distribution)
        md = mahalanobis_sq(data[["u", "w"]].to_numpy(), res[["x", "y"]].to_numpy())
        np.testing.assert_allclose(md, quantile(distribution, level, 40, 2))

    def test_center_is_the_mean(self, data):
        # theta runs over whole degrees, so the 360 distinct points are symmetric about the center
        res = confidence_ellipse(data, x="u", y="v")
        np.testing.assert_allclose(res.iloc[:-1].mean(), data[["u", "v"]].mean())

    @pytest.mark.parametrize("distribution", ["normal", "hotelling"])
    def test_semi_axes_and_orientation(self, data, distribution):
        x = data[["u", "v"]].to_numpy()
        # Sample covariance matrix and its eigenvalues, in closed form for a 2 x 2 matrix
        xc = x - x.mean(axis=0)
        (s11, s12), (_, s22) = xc.T @ xc / (len(x) - 1)
        mid, half_gap = (s11 + s22) / 2, np.hypot((s11 - s22) / 2, s12)
        major_angle = np.arctan2(2 * s12, s11 - s22) / 2
        c = quantile(distribution, 0.95, 40, 2)

        res = confidence_ellipse(data, x="u", y="v", distribution=distribution)
        pts = res.to_numpy() - x.mean(axis=0)
        r = np.hypot(pts[:, 0], pts[:, 1])
        # The ends of both axes are among the points (theta = 0, 90, 180 and 270 degrees)
        assert r.max() == pytest.approx(np.sqrt(c * (mid + half_gap)))
        assert r.min() == pytest.approx(np.sqrt(c * (mid - half_gap)))
        # The farthest points are on the major axis
        far = pts[np.argmax(r)]
        assert np.sin(np.arctan2(far[1], far[0]) - major_angle) == pytest.approx(0, abs=1e-8)

    def test_hotelling_ellipse_matches_hotelling_parameters(self, data):
        # hotelling_parameters is checked against the R package HotellingEllipse
        x = data[["u", "v"]].to_numpy()
        params = hotelling_parameters(x, conf_limit=0.99)
        res = confidence_ellipse(data, x="u", y="v", conf_level=0.99, distribution="hotelling")
        np.testing.assert_allclose(mahalanobis_sq(x, res.to_numpy()), params["cutoff_99pct"])
        pts = res.to_numpy() - x.mean(axis=0)
        r = np.hypot(pts[:, 0], pts[:, 1])
        a, b = params["Ellipse"]["a_99pct"][0], params["Ellipse"]["b_99pct"][0]
        assert r.max() == pytest.approx(max(a, b))
        assert r.min() == pytest.approx(min(a, b))


class TestEllipsoid:
    @pytest.mark.parametrize("distribution", ["normal", "hotelling"])
    @pytest.mark.parametrize("level", [0.5, 0.95, 0.999])
    def test_points_lie_on_the_quantile(self, data, distribution, level):
        res = confidence_ellipse(data, x="v", y="w", z="u", conf_level=level, distribution=distribution)
        md = mahalanobis_sq(data[["v", "w", "u"]].to_numpy(), res[["x", "y", "z"]].to_numpy())
        np.testing.assert_allclose(md, quantile(distribution, level, 40, 3))

    @pytest.mark.parametrize("distribution", ["normal", "hotelling"])
    def test_points_reach_the_ends_of_the_principal_axes(self, data, distribution):
        x = data[["u", "v", "w"]].to_numpy()
        values, vectors = np.linalg.eigh(np.cov(x, rowvar=False))
        c = quantile(distribution, 0.95, 40, 3)
        res = confidence_ellipse(data, x="u", y="v", z="w", distribution=distribution)
        # Coordinates of the points along the principal axes: the 50 x 50 grid reaches
        # the ends of the semi-axes up to its spacing
        proj = (res.to_numpy() - x.mean(axis=0)) @ vectors
        np.testing.assert_allclose(np.abs(proj).max(axis=0), np.sqrt(c * values), rtol=2e-3)


class TestSingularCovariance:
    def test_collinear_variables_give_a_line_segment(self, data):
        # Rounding makes the zero eigenvalue of the covariance matrix slightly negative
        data["v"] = 2 * data["u"]
        res = confidence_ellipse(data, x="u", y="v")
        assert np.isfinite(res.to_numpy()).all()
        np.testing.assert_allclose(res["y"], 2 * res["x"])
        # The covariance matrix is var(u) [[1, 2], [2, 4]], with eigenvalues 0 and 5 var(u)
        r = np.hypot(*(res.to_numpy() - data[["u", "v"]].mean().to_numpy()).T)
        assert r.max() == pytest.approx(np.sqrt(stats.chi2.ppf(0.95, 2) * 5 * data["u"].var()))

    def test_collinear_variables_give_a_flat_ellipsoid(self, data):
        data["w"] = data["u"] - data["v"]
        res = confidence_ellipse(data, x="u", y="v", z="w")
        assert np.isfinite(res.to_numpy()).all()
        np.testing.assert_allclose(res["z"], res["x"] - res["y"], atol=1e-12)


class TestGroups:
    @pytest.mark.parametrize("distribution", ["normal", "hotelling"])
    @pytest.mark.parametrize("z", [None, "w"])
    def test_each_group_has_its_own_region(self, grouped, z, distribution):
        coords = ["x", "y", "z"][: len(variables(z))]
        pts_per_group = 361 if z is None else 50 * 50
        res = confidence_ellipse(grouped, x="u", y="v", z=z, group_by="group", conf_level=0.9, distribution=distribution)
        assert list(res.columns) == coords + ["group"]
        assert len(res) == 2 * pts_per_group
        pd.testing.assert_index_equal(res.index, pd.RangeIndex(len(res)))
        for name, n in [("A", 12), ("B", 28)]:
            x = grouped.loc[grouped["group"] == name, variables(z)].to_numpy()
            pts = res.loc[res["group"] == name, coords].to_numpy()
            assert len(pts) == pts_per_group
            # Mean, covariance matrix and number of observations of the group only
            np.testing.assert_allclose(mahalanobis_sq(x, pts), quantile(distribution, 0.9, n, len(variables(z))))

    @pytest.mark.parametrize("z, n_min", [(None, 3), ("w", 4)])
    def test_group_with_too_few_observations(self, grouped, z, n_min):
        small = pd.concat([grouped[grouped["group"] == "B"], grouped[grouped["group"] == "A"].head(n_min - 1)])
        with pytest.raises(ValueError, match=f"At least {n_min} observations are required."):
            confidence_ellipse(small, x="u", y="v", z=z, group_by="group")

    # pandas < 3 warns that the default of observed will change
    @pytest.mark.filterwarnings("error::FutureWarning")
    def test_unused_categories_are_ignored(self, grouped):
        # e.g. after removing the observations of a group from data
        grouped["group"] = pd.Categorical(grouped["group"], categories=["A", "B", "C"])
        res = confidence_ellipse(grouped, x="u", y="v", group_by="group")
        assert set(res["group"]) == {"A", "B"}
        assert len(res) == 2 * 361


class TestRobust:
    @pytest.mark.parametrize("distribution", ["normal", "hotelling"])
    @pytest.mark.parametrize("z", [None, "w"])
    def test_estimates_are_the_reweighted_mcd(self, contaminated, z, distribution):
        x = contaminated[variables(z)].to_numpy()
        location, covariance = reweighted_mcd(x)
        res = confidence_ellipse(contaminated, x="u", y="v", z=z, robust=True, distribution=distribution)
        d = res.to_numpy() - location
        md = np.einsum("ij,jk,ik->i", d, np.linalg.inv(covariance), d)
        np.testing.assert_allclose(md, quantile(distribution, 0.95, 200, len(variables(z))))

    @pytest.mark.skipif(SKLEARN_BEFORE_1_8, reason="scikit-learn < 1.8 does not apply the consistency factor")
    @pytest.mark.parametrize("z", [None, "w"])
    def test_consistency_factor_is_the_one_of_scikit_learn(self, contaminated, z):
        x = contaminated[variables(z)].to_numpy()
        mcd = MinCovDet(support_fraction=0.9, random_state=42).fit(x)
        location, covariance = reweighted_mcd(x)
        np.testing.assert_allclose(location, mcd.location_)
        np.testing.assert_allclose(covariance, mcd.covariance_)

    @pytest.mark.parametrize("z", [None, "w"])
    def test_region_resists_outliers(self, contaminated, z):
        clean = contaminated[variables(z)].to_numpy()[10:]
        c = stats.chi2.ppf(0.95, len(variables(z)))

        def relative_distance(robust):
            """Squared Mahalanobis distance of the points relative to the 95% region of the clean observations."""
            res = confidence_ellipse(contaminated, x="u", y="v", z=z, robust=robust)
            return mahalanobis_sq(clean, res.to_numpy()) / c

        robust, classical = relative_distance(True), relative_distance(False)
        assert robust.min() > 0.5 and robust.max() < 1.5
        assert classical.max() > 2

    def test_factor_is_applied_with_scikit_learn_before_1_8(self, contaminated, monkeypatch):
        # Whichever version is installed: the covariance matrix of scikit-learn is multiplied
        # by the factor
        monkeypatch.setattr(confidence_ellipse_module, "_sklearn_version", lambda: (1, 7))
        x = contaminated[["u", "v"]].to_numpy()
        mcd = MinCovDet(support_fraction=0.9, random_state=42).fit(x)
        factor = 0.975 / stats.chi2.cdf(stats.chi2.ppf(0.975, 2), 4)
        res = confidence_ellipse(contaminated, x="u", y="v", robust=True)
        d = res.to_numpy() - mcd.location_
        md = np.einsum("ij,jk,ik->i", d, np.linalg.inv(mcd.covariance_ * factor), d)
        np.testing.assert_allclose(md, stats.chi2.ppf(0.95, 2))

    @pytest.mark.parametrize(
        "version, expected", [("1.7.2", (1, 7)), ("1.8.0", (1, 8)), ("1.10.0rc1", (1, 10)), ("2.0.dev0", (2, 0))]
    )
    def test_sklearn_version(self, monkeypatch, version, expected):
        monkeypatch.setattr(sklearn, "__version__", version)
        assert confidence_ellipse_module._sklearn_version() == expected

    @pytest.mark.parametrize("z", [None, "w"])
    def test_region_has_the_nominal_level_on_normal_data(self, z):
        # Without outliers, the robust and classical estimates agree on average: the points
        # lie on the region given by the sample mean and covariance matrix. Without the
        # consistency factor (scikit-learn < 1.8), the variances are about 10% too small, and
        # the 95% region contains about 93% of the distribution.
        rng = np.random.default_rng(1)
        c = stats.chi2.ppf(0.95, len(variables(z)))
        ratios = []
        for _ in range(10):
            df = pd.DataFrame(rng.standard_normal((2000, 3)) @ MIXING.T, columns=["u", "v", "w"])
            res = confidence_ellipse(df, x="u", y="v", z=z, robust=True)
            ratios.append(np.mean(mahalanobis_sq(df[variables(z)].to_numpy(), res.to_numpy())) / c)
        assert np.mean(ratios) == pytest.approx(1, abs=0.03)

    @pytest.mark.parametrize("z", [None, "w"])
    def test_falls_back_to_classical_estimates(self, z):
        # e.g. values at a detection limit: 46 of the 50 observations are tied, so the MCD
        # covariance matrix of the support is zero and scikit-learn raises an error
        x = np.zeros((50, 3))
        x[46:] = np.random.default_rng(0).standard_normal((4, 3))
        df = pd.DataFrame(x, columns=["u", "v", "w"])
        with pytest.warns(UserWarning, match="Robust estimation failed: .*Using classical estimates."):
            res = confidence_ellipse(df, x="u", y="v", z=z, robust=True)
        pd.testing.assert_frame_equal(res, confidence_ellipse(df, x="u", y="v", z=z))

    def test_missing_values(self, data):
        data.iloc[5, 1] = np.nan
        with pytest.warns(UserWarning, match="Robust estimation failed"):
            with pytest.raises(ValueError, match="Covariance matrix contains NA values."):
                confidence_ellipse(data, x="u", y="v", robust=True)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, error, message",
        [
            ({"x": "a"}, ValueError, "Column 'a' not found in data."),
            ({"y": "a"}, ValueError, "Column 'a' not found in data."),
            ({"z": "a"}, ValueError, "Column 'a' not found in data."),
            ({"group_by": "a"}, ValueError, "Column 'a' not found in data."),
            ({"z": "w", "group_by": "a"}, ValueError, "Column 'a' not found in data."),
            ({"conf_level": "0.95"}, TypeError, "'conf_level' must be numeric."),
            ({"conf_level": None}, TypeError, "'conf_level' must be numeric."),
            ({"conf_level": [0.95]}, TypeError, "'conf_level' must be numeric."),
            ({"conf_level": 0}, ValueError, "'conf_level' must be between 0 and 1."),
            ({"conf_level": 1}, ValueError, "'conf_level' must be between 0 and 1."),
            ({"conf_level": -0.05}, ValueError, "'conf_level' must be between 0 and 1."),
            ({"conf_level": 95}, ValueError, "'conf_level' must be between 0 and 1."),
            ({"conf_level": True}, TypeError, "'conf_level' must be numeric."),
            ({"conf_level": np.inf}, ValueError, "'conf_level' must be between 0 and 1."),
            ({"conf_level": np.nan}, ValueError, "'conf_level' must be between 0 and 1."),
            ({"distribution": "chisq"}, ValueError, "'distribution' must be either 'normal' or 'hotelling'."),
            ({"distribution": "Hotelling"}, ValueError, "'distribution' must be either 'normal' or 'hotelling'."),
        ],
    )
    def test_invalid_arguments(self, data, kwargs, error, message):
        with pytest.raises(error, match=re.escape(message)):
            confidence_ellipse(data, **{"x": "u", "y": "v", **kwargs})

    @pytest.mark.parametrize(
        "bad", [np.ones((40, 3)), {"u": [1.0, 2.0, 3.0], "v": [3.0, 1.0, 2.0]}, None], ids=["array", "dict", "None"]
    )
    def test_invalid_data_type(self, bad):
        with pytest.raises(TypeError, match="Input 'data' must be a pandas DataFrame."):
            confidence_ellipse(bad, x="u", y="v")

    @pytest.mark.parametrize("distribution", ["normal", "hotelling"])
    @pytest.mark.parametrize("z, n_min", [(None, 3), ("w", 4)])
    def test_too_few_observations(self, data, z, n_min, distribution):
        # p + 1 observations of p variables: fewer give a singular covariance matrix, and
        # n - p = 0 in the Hotelling quantile
        with pytest.raises(ValueError, match=f"At least {n_min} observations are required."):
            confidence_ellipse(data.head(n_min - 1), x="u", y="v", z=z, distribution=distribution)
        res = confidence_ellipse(data.head(n_min), x="u", y="v", z=z, distribution=distribution)
        assert np.isfinite(res.to_numpy()).all()

    @pytest.mark.parametrize("z", [None, "w"])
    def test_missing_values(self, data, z):
        data.iloc[5, 1] = np.nan
        with pytest.raises(ValueError, match="Covariance matrix contains NA values."):
            confidence_ellipse(data, x="u", y="v", z=z)
