from pathlib import Path
import warnings

import numpy as np
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
        "coverage": float(FIXTURE["cirCoverage"].item()),
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


def test_cir_grid_methods_match_matlab():
    parameters = _parameters()
    for method, grid_name, transition_name in (
        ("exponential", "cirGridExponential", "cirTransitionExponential"),
        ("even", "cirGridEven", "cirTransitionEven"),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            transition, grid = discrete_cir(method=method, **parameters)

        np.testing.assert_allclose(grid, FIXTURE[grid_name].reshape(-1), atol=1e-12)
        np.testing.assert_allclose(
            transition, FIXTURE[transition_name], atol=1e-8, rtol=0
        )
        np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(transition >= 0)
