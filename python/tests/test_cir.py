from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from discretization import cir_transition_density, discrete_cir

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = loadmat(REPOSITORY_ROOT / "reference_data" / "matlab_r2024b_milestone2.mat")


def _parameters():
    return {
        "mean_reversion": float(FIXTURE["cirA"].item()),
        "long_run_mean": float(FIXTURE["cirB"].item()),
        "volatility": float(FIXTURE["cirSigma"].item()),
        "step": float(FIXTURE["cirDelta"].item()),
        "state_count": int(FIXTURE["cirStateCount"].item()),
    }


def test_cir_density_matches_matlab():
    parameters = _parameters()
    density = cir_transition_density(
        FIXTURE["cirDensityGrid"].reshape(-1),
        float(FIXTURE["cirDensityCurrent"].item()),
        parameters["mean_reversion"],
        parameters["long_run_mean"],
        parameters["volatility"],
        parameters["step"],
    )

    np.testing.assert_allclose(
        density, FIXTURE["cirDensity"].reshape(-1), atol=1e-11, rtol=1e-11
    )


def test_cir_explicit_even_uses_corrected_moments():
    parameters = _parameters()
    p, x = discrete_cir(**parameters)
    explicit, y = discrete_cir(method="even", **parameters)
    np.testing.assert_array_equal(x, y)
    np.testing.assert_allclose(p, explicit, atol=1e-12, rtol=0)
    a, b, s, gap = (
        parameters[k] for k in ("mean_reversion", "long_run_mean", "volatility", "step")
    )
    rho = np.exp(-a * gap)
    c = -np.expm1(-a * gap)
    mean = rho * x + c * b
    variance = s * s / a * c * (rho * x + b * c / 2)
    np.testing.assert_allclose((p @ x - mean) / np.sqrt(variance), 0, atol=1e-7, rtol=0)
    np.testing.assert_allclose(
        np.sum(p * (x - mean[:, None]) ** 2, axis=1), variance, atol=0, rtol=1e-7
    )


def test_retired_cir_interfaces_are_not_silent_legacy_paths():
    with pytest.raises(ValueError, match="only method='even'"):
        discrete_cir(method="exponential", **_parameters())
    with pytest.raises(TypeError, match="coverage"):
        discrete_cir(coverage=0.999, **_parameters())
