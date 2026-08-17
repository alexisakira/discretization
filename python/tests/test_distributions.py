from pathlib import Path

import numpy as np
from scipy.io import loadmat

from discretization import moment_matched_distribution

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = loadmat(REPOSITORY_ROOT / "reference_data" / "matlab_r2024b_milestone2.mat")


def test_moment_matched_distribution_matches_matlab():
    grid, probability = moment_matched_distribution(
        int(FIXTURE["discreteNPStateCount"].item()),
        FIXTURE["discreteNPMoments"].reshape(-1),
    )

    np.testing.assert_allclose(grid, FIXTURE["discreteNPGrid"].reshape(-1), atol=1e-12)
    np.testing.assert_allclose(
        probability, FIXTURE["discreteNPProbability"].reshape(-1), atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(probability.sum(), 1.0, atol=1e-12)
