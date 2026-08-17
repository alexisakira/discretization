from pathlib import Path

import numpy as np
from scipy.io import loadmat

from discretization import entropy_objective, maximum_entropy_weights

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = loadmat(REPOSITORY_ROOT / "reference_data" / "matlab_r2024b_baseline.mat")


def test_maximum_entropy_matches_matlab_reference():
    grid = FIXTURE["coreGrid"].reshape(-1)
    targets = FIXTURE["coreTargetMoments"].reshape(-1)

    probability, _, moment_error = maximum_entropy_weights(
        grid, lambda values: np.vstack((values, values**2)), targets
    )

    np.testing.assert_allclose(
        probability, FIXTURE["coreProbability"].reshape(-1), atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        moment_error, FIXTURE["coreMomentError"].reshape(-1), atol=1e-8, rtol=0
    )


def test_log_sum_exp_objective_remains_finite():
    objective, gradient, hessian, probability = entropy_objective(
        [1000.0], [[1000.0, -1000.0]], [0.0], [0.5, 0.5]
    )

    assert np.isfinite(objective)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    np.testing.assert_allclose(objective, 1e6 - np.log(2), atol=1e-8)
    np.testing.assert_allclose(probability, [1.0, 0.0], atol=np.finfo(float).tiny)
