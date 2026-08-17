from pathlib import Path

import numpy as np
from scipy.io import loadmat

from discretization import discrete_ar

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = loadmat(REPOSITORY_ROOT / "reference_data" / "matlab_r2024b_baseline.mat")


def test_discrete_ar_matches_matlab_reference():
    transition, grid = discrete_ar(
        mean=float(FIXTURE["arMu"].item()),
        persistence=float(FIXTURE["arRho"].item()),
        innovation_std=float(FIXTURE["arSigma"].item()),
        state_count=int(FIXTURE["arStateCount"].item()),
        method=str(FIXTURE["arMethod"].item()),
        moment_count=int(FIXTURE["arMomentCount"].item()),
    )

    np.testing.assert_allclose(grid, FIXTURE["arGrid"].reshape(-1), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        transition, FIXTURE["arTransition"], atol=1e-7, rtol=0
    )
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(transition >= 0)

