"""Independent regression checks for the final numerical review cases."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.stats import ncx2

from discretization import (
    cir_transition_density,
    discrete_cir,
    discrete_gaussian_mixture_ar,
    discrete_stochastic_volatility_ar,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("gap", [1, 5, 10, 20, 1000])
def test_cir_log_density_against_independent_ncx2_pdf(gap):
    # SciPy's Boost-based pdf is independent of its underflow-prone logpdf.
    x = np.array([0.9, 1.0, 1.1])
    c = 2 / (-np.expm1(-gap) * 0.1**2)
    expected = 2 * c * ncx2.pdf(2 * c * x, 400, 2 * c * np.exp(-gap))
    actual = cir_transition_density(x, 1, 1, 1, 0.1, gap)
    assert (actual > 0).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0)


@pytest.mark.parametrize("gap", [10, 1000])
def test_cir_long_gap_chain_moments_and_matlab_parity(gap):
    p, x = discrete_cir(1, 1, 0.1, gap, 51)
    rho = np.exp(-gap)
    c = -np.expm1(-gap)
    mean = rho * x + c
    variance = 0.01 * c * (rho * x + c / 2)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose((p @ x - mean) / np.sqrt(variance), 0, atol=1e-7, rtol=0)
    np.testing.assert_allclose(
        np.sum(p * (x - mean[:, None]) ** 2, axis=1) / variance, 1, atol=1e-7, rtol=0
    )
    fixture = loadmat(ROOT / "reference_data/matlab_r2026a_0_2.mat")
    np.testing.assert_allclose(x, fixture[f"cirGap{gap}X"].ravel(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(p, fixture[f"cirGap{gap}P"], atol=1e-7, rtol=0)


@pytest.mark.parametrize(
    "method",
    ["even", "gauss_legendre", "clenshaw_curtis", "gauss_hermite", "gaussian_mixture"],
)
@pytest.mark.parametrize("moments", [2, 4])
def test_all_mixture_grids_location_invariant(method, moments):
    args = ([0.4], [0.4, 0.6], [-0.1, 0.1], [0.1, 0.2], 21)
    p, x, d = discrete_gaussian_mixture_ar(
        0,
        *args,
        moment_count=moments,
        method=method,
        grid_width=4,
        return_diagnostics=True,
    )
    for location in (0.5, 10):
        q, y, info = discrete_gaussian_mixture_ar(
            location,
            *args,
            moment_count=moments,
            method=method,
            grid_width=4,
            return_diagnostics=True,
        )
        np.testing.assert_allclose(y - x, location, atol=1e-12, rtol=0)
        np.testing.assert_allclose(q, p, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(info["matched_moments"], d["matched_moments"])
    if method in ("gauss_hermite", "gaussian_mixture"):
        fixture = loadmat(ROOT / "reference_data/matlab_r2026a_0_2.mat")
        index = 4 if method == "gauss_hermite" else 5
        np.testing.assert_allclose(
            q, fixture[f"gmShiftP{index}M{moments}"], atol=1e-7, rtol=0
        )


def test_sv_exposes_nested_fallback_and_matches_matlab():
    with pytest.warns(RuntimeWarning, match="quantile"):
        p, x, d = discrete_stochastic_volatility_ar(
            0, 0.99, 0.007, 0.06, 9, 3, method="quantile", return_diagnostics=True
        )
    np.testing.assert_array_equal(d["matched_moments"], np.full(27, 2))
    assert d["downgraded_rows"].size == 0
    np.testing.assert_array_equal(d["volatility"]["matched_moments"].ravel(), [1, 2, 1])
    np.testing.assert_array_equal(d["volatility"]["downgraded_rows"], [0, 2])
    fixture = loadmat(ROOT / "reference_data/matlab_r2026a_0_2.mat")
    np.testing.assert_allclose(x, fixture["svFallbackX"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(p, fixture["svFallbackP"], atol=1e-7, rtol=0)
