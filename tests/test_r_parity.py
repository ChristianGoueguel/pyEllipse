"""
Parity with the HotellingEllipse 1.3.0 R package.

The reference values below come from tests/r_parity_reference.R, which runs
ellipseParam() and ellipseCoord() on DATA. The columns are correlated, so the
rotated ellipse (ellipsoid) is exercised as well.
"""
import numpy as np
import pytest

from pyEllipse import hotelling_coordinates, hotelling_parameters

DATA = np.array([
    [-2.38, -0.59, -2.49, 0.28],
    [1.91, 0.33, 0.24, 0.13],
    [-0.80, -0.66, 0.77, 0.16],
    [-0.19, -0.21, 0.20, -0.27],
    [-1.21, 0.34, -0.77, -0.48],
    [-1.43, 0.41, -1.00, 0.02],
    [1.93, 3.51, -1.85, 1.14],
    [-3.69, -1.21, -1.92, 0.23],
    [2.50, 2.71, -1.38, 0.69],
    [-1.56, -1.31, 0.61, 0.16],
    [0.61, -0.70, -0.22, 0.35],
    [1.78, 1.79, 1.90, 0.23],
])

# Output of `Rscript tests/r_parity_reference.R`
PARAMETERS = {
    'k2_pcx1_pcy3': {
        'Tsquared': [2.7617873763507306, 1.2128236134866106, 1.3683544780781478, 0.31332075485157951, 0.26069274328396386, 0.42370009745097637, 3.3795579831480964, 3.4316897123328589, 3.3390417358101576, 1.7577120125231978, 0.18021424585147247, 3.5711052468322073],
        'cutoff_99pct': 16.630750746605379,
        'cutoff_95pct': 9.0262062332868815,
        'nb_comp': 2,
        'Ellipse': {'a_99pct': 8.3088948579238924, 'b_99pct': 4.8381761016435441, 'a_95pct': 6.121247817118884, 'b_95pct': 3.5643338142351015, 'angle': 0.32809981638158381},
    },
    'k2_pcx2_pcy1_beta_levels': {
        'Tsquared': [1.3293095867175881, 2.8537869611561408, 0.54607025148136845, 0.34728457837317556, 0.58017809335254655, 0.97475560098524416, 4.6603938058066845, 3.3766335985379445, 2.4273525021231399, 1.2315758784640696, 2.592082867198088, 1.0805762758040107],
        'cutoff_97.5pct': 5.6412081536401271,
        'cutoff_90pct': 4.0385338092840115,
        'nb_comp': 2,
        'Ellipse': {'a_97.5pct': 1.9859420189817949, 'b_97.5pct': 5.59207338503583, 'a_90pct': 1.6803214208055581, 'b_90pct': 4.7314980023485216, 'angle': -0.62906208124486818},
    },
    'k3': {
        'Tsquared': [3.0284585967077517, 3.3447087325030145, 1.3683889959480258, 0.41327720297247983, 0.71337032428621394, 1.0840025464987659, 4.8633856352905322, 3.4747937932637845, 3.3390704744397888, 1.8141776379117807, 3.7463047477837965, 5.8100613123940716],
        'cutoff_99pct': 25.637029814856028,
        'cutoff_95pct': 14.162677311290803,
        'nb_comp': 3,
    },
    'threshold_0.9_beta': {
        'Tsquared': [3.0284585967077517, 3.3447087325030145, 1.3683889959480258, 0.41327720297247983, 0.71337032428621394, 1.0840025464987659, 4.8633856352905322, 3.4747937932637845, 3.3390704744397888, 1.8141776379117807, 3.7463047477837965, 5.8100613123940716],
        'cutoff_99pct': 7.4619857237049532,
        'cutoff_95pct': 6.0896456677449455,
        'nb_comp': 3,
    },
}

COORD_2D = {'x': [4.2715333460609788, 1.3686220877239834, -2.9178058689593032, -2.6640527781803409, 1.7792032133546813, 4.2715333460609788],
            'y': [-0.70835527179295288, 2.5723799004183205, 1.6175552216878686, -2.2532940552760232, -3.6907857950372116, -0.70835527179295366]}

COORD_3D = {'x': [0.060448077474450163, 0.060448077474450163, 0.060448077474450163, 0.060448077474450163, 3.8290432439313857, -0.69688543128094116, -3.3577356964387697, 3.8290432439313848, 3.5577618331236027, -0.96816684208872483, -3.6290171072465536, 3.5577618331236023, -0.48211474414111621, -0.48211474414111688, -0.48211474414111721, -0.48211474414111621],
            'y': [0.81250086796450494, 0.81250086796450494, 0.81250086796450494, 0.81250086796450494, 2.1262430508444106, 2.323495358052543, -2.6797371069501952, 2.1262430508444097, 1.6812421828799058, 1.8784944900880383, -3.1247379749146997, 1.6812421828799049, -0.077500867964504727, -0.077500867964504672, -0.077500867964505393, -0.077500867964504727],
            'z': [1.0949300779451467, 1.0949300779451467, 1.0949300779451467, 1.0949300779451467, 0.8924016323065963, 0.87374739327894069, 0.20624609133218347, 0.89240163230659619, 0.017471554361449565, -0.0011826846662059132, -0.66868398661296335, 0.017471554361449482, -0.65493007794514679, -0.65493007794514679, -0.65493007794514679, -0.65493007794514679]}


CALLS = {
    "k2_pcx1_pcy3": dict(pcx=1, pcy=3),
    "k2_pcx2_pcy1_beta_levels": dict(pcx=2, pcy=1, method="beta", conf_limit=(0.9, 0.975)),
    "k3": dict(k=3),
    "threshold_0.9_beta": dict(threshold=0.9, method="beta"),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_hotelling_parameters_matches_r(name):
    res, expected = hotelling_parameters(DATA, **CALLS[name]), PARAMETERS[name]
    assert list(res) == list(expected)
    np.testing.assert_allclose(res["Tsquared"]["value"], expected["Tsquared"], rtol=1e-10)
    for key in expected:
        if key.startswith("cutoff"):
            assert res[key] == pytest.approx(expected[key], rel=1e-10)
    assert res["nb_comp"] == expected["nb_comp"]
    if "Ellipse" in expected:
        assert list(res["Ellipse"].columns) == list(expected["Ellipse"])
        for column, value in expected["Ellipse"].items():
            assert res["Ellipse"][column][0] == pytest.approx(value, rel=1e-10)


def test_hotelling_coordinates_2d_matches_r():
    res = hotelling_coordinates(DATA, pcx=2, pcy=3, conf_limit=0.9, pts=6)
    for column, values in COORD_2D.items():
        np.testing.assert_allclose(res[column], values, rtol=1e-10, atol=1e-12)


def test_hotelling_coordinates_3d_matches_r():
    res = hotelling_coordinates(DATA, pcx=1, pcy=2, pcz=4, pts=4, method="beta")
    for column, values in COORD_3D.items():
        np.testing.assert_allclose(res[column], values, rtol=1e-10, atol=1e-12)
