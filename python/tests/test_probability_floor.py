from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from discretization import (
    discrete_ar,
    discrete_cir,
    discrete_gaussian_mixture_ar,
    discrete_ou,
    discrete_stochastic_volatility_ar,
    discrete_var,
)

ROOT = Path(__file__).resolve().parents[2]


def test_gaussian_tails_density_and_diagnostics():
    p, x, d = discrete_ar(0, 0.9, 0.1, 81, return_diagnostics=True)
    z = (x - 0.9 * x[:, None]) / 0.1
    assert np.max(p[np.abs(z) > 10]) < 1e-15
    assert np.all(d["matched_moments"] == 2)
    q = np.exp(-0.5 * z * z)
    q /= q.sum(axis=1)[:, None]
    interior = np.minimum((x[-1] - 0.9 * x) / 0.1, (0.9 * x - x[0]) / 0.1) > 8
    assert interior.any()
    np.testing.assert_allclose(p[interior], q[interior], atol=1e-11, rtol=0)
    assert np.abs(d["standardized_moment_errors"][:, :2]).max() <= d["moment_tolerance"]
    legacy, _ = discrete_ar(0, 0.9, 0.1, 81, probability_floor=1e-8)
    assert np.max(legacy[np.abs(z) > 10]) > 1e-12


@pytest.mark.parametrize("name", ["ar", "var", "gmar", "sv"])
def test_explicit_floor_reproduces_legacy_fixtures(name):
    f = loadmat(ROOT / "reference_data/matlab_r2024b_performance.mat")
    calls = {
        "ar": (
            lambda: discrete_ar(0, 0.9, 0.1, 9, moment_count=4, probability_floor=1e-8),
            "ar4",
        ),
        "var": (
            lambda: discrete_var(
                [0, 0],
                [[0.9809, 0.0028], [0.0410, 0.9648]],
                np.diag([0.0087**2, 0.0262**2]),
                9,
                probability_floor=1e-8,
            ),
            "var",
        ),
        "gmar": (
            lambda: discrete_gaussian_mixture_ar(
                0.0555,
                [0.5854],
                [0.1628, 0.8372],
                [-0.0039, 0.0008],
                [0.1293, 0.0300],
                9,
                probability_floor=1e-8,
            ),
            "gmar1",
        ),
        "sv": (
            lambda: discrete_stochastic_volatility_ar(
                0.95, 0.9, 0.007, 0.06, 9, 5, probability_floor=1e-8
            ),
            "sv",
        ),
    }
    call, key = calls[name]
    p, x = call()
    if name == "ar":
        # The 0.1.0 Python/MATLAB four-moment solvers already differed by
        # 3.3e-6. Check numerical reproduction of the original Python output,
        # as well as its historical cross-language accuracy.
        old = np.load(ROOT / "reference_data/python_0_1_ar4.npz")
        np.testing.assert_allclose(p, old["transition"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(x, old["grid"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(p, f[key + "Transition"], atol=4e-6, rtol=0)
    else:
        np.testing.assert_allclose(p, f[key + "Transition"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(
        x,
        np.asarray(f[key + "Grid"]).squeeze() if name == "ar" else f[key + "Grid"],
        atol=1e-8,
        rtol=0,
    )


@pytest.mark.parametrize(
    "method",
    ["even", "gauss_legendre", "clenshaw_curtis", "gauss_hermite", "gaussian_mixture"],
)
def test_mixture_diagnostics(method):
    p, x, d = discrete_gaussian_mixture_ar(
        0,
        [0.4],
        [0.4, 0.6],
        [-0.1, 0.1],
        [0.1, 0.2],
        21,
        4,
        method,
        4,
        return_diagnostics=True,
    )
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
    fixture = loadmat(
        ROOT / "reference_data/matlab_r2026a_0_2.mat", simplify_cells=True
    )
    index = [
        "even",
        "gauss_legendre",
        "clenshaw_curtis",
        "gauss_hermite",
        "gaussian_mixture",
    ].index(method) + 1
    np.testing.assert_allclose(p, fixture[f"gmOptionalP{index}"], atol=1e-7, rtol=0)
    # High-order mixture Gaussian quadrature has a pre-existing eigensolver
    # sensitivity across languages. Its nodes agree to 3e-8 at N=21; other
    # grid constructions agree to 1e-12. Transition parity stays at 1e-7.
    grid_tolerance = 3e-8 if method == "gaussian_mixture" else 1e-12
    np.testing.assert_allclose(
        x, fixture[f"gmOptionalX{index}"][None, :], atol=grid_tolerance, rtol=0
    )
    np.testing.assert_array_equal(
        d["matched_moments"], fixture[f"gmOptionalD{index}"]["matchedMoments"]
    )
    np.testing.assert_allclose(
        d["innovation_sd"], fixture[f"gmOptionalD{index}"]["innovationSD"], atol=1e-14
    )
    np.testing.assert_allclose(
        d["unconditional_mean"],
        fixture[f"gmOptionalD{index}"]["unconditionalMean"],
        atol=1e-14,
    )
    for row, matched in enumerate(d["matched_moments"]):
        if matched:
            assert (
                np.abs(d["standardized_moment_errors"][row, :matched]).max()
                <= d["moment_tolerance"]
            )


def test_infeasible_means_and_variances():
    p, x, d = discrete_gaussian_mixture_ar(
        0, [0.4], [1], [10], [0.1], 9, return_diagnostics=True
    )
    assert (d["matched_moments"] == 0).all()
    assert (p[:, -1] == 1).all()
    p, x, d = discrete_gaussian_mixture_ar(
        0, [0.4], [1], [0], [0.1], 9, grid_width=0.5, return_diagnostics=True
    )
    assert (d["matched_moments"] == 1).all()
    assert np.abs(d["standardized_moment_errors"][:, 0]).max() <= 1e-7


def test_ar2_lags_and_matlab_diagnostic_parity():
    f = loadmat(ROOT / "reference_data/matlab_r2026a_0_2.mat", simplify_cells=True)
    p, x, d = discrete_gaussian_mixture_ar(
        0.0555,
        [0.8959, -0.3990],
        [0.1628, 0.8372],
        [-0.0039, 0.0008],
        [0.1293, 0.0300],
        9,
        return_diagnostics=True,
    )
    np.testing.assert_array_equal(
        d["matched_moments"], f["gmar2Diagnostics"]["matchedMoments"]
    )
    for row in range(x.shape[1]):
        assert np.all(p[row, x[1] != x[0, row]] == 0)


@pytest.mark.parametrize("model", ["cir", "ou"])
def test_new_diffusions_match_matlab(model):
    f = loadmat(ROOT / "reference_data/matlab_r2026a_0_2.mat")
    if model == "cir":
        p, x = discrete_cir(-np.log(0.58), 0.028, 0.1, 0.25, 9)
    else:
        p, x = discrete_ou(0.7, 0.3, 0.25, 21)
    np.testing.assert_allclose(x, f[model + "Grid"].ravel(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(p, f[model + "Transition"], atol=1e-7, rtol=0)


@pytest.mark.parametrize("gap", [0.25, 0.01, 0.001, 0.0001])
def test_ou_moments_fixed_and_infeasible_grid(gap):
    p, x = discrete_ou(0.7, 0.3, gap, 21)
    rho = np.exp(-0.7 * gap)
    variance = 0.3**2 / (2 * 0.7) * (-np.expm1(-2 * 0.7 * gap))
    np.testing.assert_allclose(p @ x, rho * x, atol=1e-8)
    np.testing.assert_allclose(
        np.sum(p * (x - rho * x[:, None]) ** 2, axis=1) / variance, 1, atol=1e-7, rtol=0
    )
    fixed, y = discrete_ou(0.7, 0.3, gap, x)
    np.testing.assert_allclose(y, x, atol=1e-14)
    np.testing.assert_allclose(fixed, p, atol=1e-12)
    with pytest.raises(ValueError, match="infeasible"):
        discrete_ou(0.7, 0.3, 0.01, [-1, 0, 1])


@pytest.mark.parametrize("floor", [-1, 2, np.nan, np.inf])
def test_invalid_floor(floor):
    with pytest.raises(ValueError, match="probability_floor"):
        discrete_ar(0, 0.9, 0.1, 9, probability_floor=floor)


def test_rare_mixture_fallback_and_invariance():
    f = loadmat(ROOT / "reference_data/matlab_r2026a_0_2.mat", simplify_cells=True)
    args = (0, [0.95], [0.05, 0.95], [-0.8, 0.8 * 0.05 / 0.95], [0.2, 0.03], 31, 4)
    p, x, d = discrete_gaussian_mixture_ar(*args, return_diagnostics=True)
    np.testing.assert_allclose(p, f["gmSkewP"], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(d["matched_moments"], f["gmSkewD"]["matchedMoments"])
    for units, location in [(1e-3, 0), (1, 10), (1e3, 0)]:
        shifted = list(args)
        shifted[0] = location
        shifted[3] = units * np.array(args[3])
        shifted[4] = units * np.array(args[4])
        q, y, info = discrete_gaussian_mixture_ar(*shifted, return_diagnostics=True)
        np.testing.assert_allclose(q, p, atol=2e-7, rtol=0)
        np.testing.assert_allclose((y - location) / units, x, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(info["matched_moments"], d["matched_moments"])


@pytest.mark.parametrize("method", ["quantile", "quadrature"])
def test_optional_var_parity(method):
    f = loadmat(ROOT / "reference_data/matlab_r2026a_0_2.mat")
    p, x, d = discrete_var(
        [0], [[0.4]], [[0.01]], 21, 4, method, return_diagnostics=True
    )
    np.testing.assert_allclose(p, f["var" + method + "P"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(x, f["var" + method + "X"], atol=1e-12, rtol=0)
    for row in range(21):
        matched = d["matched_moments"][row, 0]
        if matched:
            assert (
                np.abs(d["standardized_moment_errors"][row, 0, :matched]).max() <= 1e-7
            )


def test_cir_feller_equality_and_infeasible_grid():
    from discretization import cir_transition_density

    # sigma^2 = 2*a*b; the density at the origin has a finite limit.
    density = cir_transition_density([0, 0.01], 0.02, 0.5, 0.0625, 0.25, 0.25)
    c = 2 * 0.5 / (-np.expm1(-0.5 * 0.25) * 0.25**2)
    np.testing.assert_allclose(
        density[0], c * np.exp(-c * 0.02 * np.exp(-0.5 * 0.25)), atol=1e-12
    )
    with pytest.raises(ValueError, match="increase state_count"):
        discrete_cir(0.5, 0.03, 0.1, 0.0001, 3)


@pytest.mark.parametrize("width", [-1, 0, np.nan, np.inf])
def test_invalid_ar_grid_width(width):
    with pytest.raises(ValueError, match="grid_width"):
        discrete_ar(0, 0.9, 0.1, 9, grid_width=width)


def test_ou_rejects_matrix_grid():
    with pytest.raises(ValueError, match="one-dimensional"):
        discrete_ou(
            0.7, 0.3, 0.25, [[-0.7, -0.5, -0.3], [-0.1, 0.1, 0.3], [0.5, 0.7, 0.9]]
        )
