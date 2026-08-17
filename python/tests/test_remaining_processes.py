from pathlib import Path
import warnings

import numpy as np
import pytest
from scipy.io import loadmat

from discretization import (
    discrete_gaussian_mixture_ar,
    discrete_stochastic_volatility_ar,
    discrete_var,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = loadmat(
    REPOSITORY_ROOT / "reference_data" / "matlab_r2024b_performance.mat"
)


def test_var_matches_matlab_reference():
    lag = np.array([[0.9809, 0.0028], [0.0410, 0.9648]])
    covariance = np.diag([0.0087**2, 0.0262**2])
    transition, grid = discrete_var(
        np.zeros(2),
        lag,
        covariance,
        9,
        moment_count=2,
        method="even",
    )

    np.testing.assert_allclose(grid, FIXTURE["varGrid"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        transition,
        FIXTURE["varTransition"],
        atol=1e-7,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("coefficients", "grid_name", "transition_name"),
    [
        ([0.5854], "gmar1Grid", "gmar1Transition"),
        ([0.8959, -0.3990], "gmar2Grid", "gmar2Transition"),
    ],
)
def test_gaussian_mixture_ar_matches_matlab_reference(
    coefficients,
    grid_name,
    transition_name,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        transition, grid = discrete_gaussian_mixture_ar(
            0.0555,
            coefficients,
            [0.1628, 0.8372],
            [-0.0039, 0.0008],
            [0.1293, 0.0300],
            9,
            moment_count=2,
            method="even",
        )

    np.testing.assert_allclose(grid, FIXTURE[grid_name], atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        transition,
        FIXTURE[transition_name],
        atol=1e-7,
        rtol=0,
    )


def test_stochastic_volatility_matches_matlab_reference():
    transition, grid = discrete_stochastic_volatility_ar(
        0.95,
        0.9,
        0.007,
        0.06,
        9,
        5,
    )

    np.testing.assert_allclose(grid, FIXTURE["svGrid"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        transition,
        FIXTURE["svTransition"],
        atol=1e-7,
        rtol=0,
    )


def test_three_dimensional_var_is_row_stochastic():
    transition, grid = discrete_var(
        np.zeros(3),
        np.array(
            [
                [0.3, 0.05, 0.0],
                [0.0, 0.4, 0.03],
                [0.02, 0.0, 0.2],
            ]
        ),
        np.array(
            [
                [1.0, 0.2, 0.1],
                [0.2, 1.5, 0.15],
                [0.1, 0.15, 0.8],
            ]
        ),
        3,
    )

    assert transition.shape == (27, 27)
    assert grid.shape == (3, 27)
    assert np.all(np.isfinite(transition))
    assert np.all(transition >= 0)
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-12)


@pytest.mark.parametrize("method", ["quantile", "quadrature"])
def test_var_additional_grid_methods(method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        transition, grid = discrete_var(0, 0.5, 0.04, 5, method=method)

    assert transition.shape == (5, 5)
    assert grid.shape == (1, 5)
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-12)


@pytest.mark.parametrize(
    "method",
    [
        "even",
        "gauss_legendre",
        "clenshaw_curtis",
        "gauss_hermite",
        "gaussian_mixture",
    ],
)
def test_gaussian_mixture_ar_grid_methods(method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        transition, grid = discrete_gaussian_mixture_ar(
            0.0,
            [0.5],
            [0.3, 0.7],
            [-0.1, 0.05],
            [0.2, 0.08],
            5,
            method=method,
        )

    assert transition.shape == (5, 5)
    assert grid.shape == (1, 5)
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-12)


def test_stochastic_volatility_accepts_quadrature_grid():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        transition, grid = discrete_stochastic_volatility_ar(
            0.5,
            0.5,
            0.1,
            0.1,
            3,
            3,
            method="quadrature",
        )

    assert transition.shape == (9, 9)
    assert grid.shape == (2, 9)
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-12)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (discrete_var, (0, 1.01, 0.04, 5)),
        (discrete_var, ([0, 0], np.eye(2), [[1, 2], [2, 1]], 3)),
        (
            discrete_gaussian_mixture_ar,
            (0, [0.5], [0.2, 0.2], [0, 0], [1, 1], 5),
        ),
        (
            discrete_gaussian_mixture_ar,
            (0, [1.1], [1], [0], [1], 5),
        ),
        (
            discrete_stochastic_volatility_ar,
            (1.0, 0.5, 0.1, 0.1, 5, 3),
        ),
    ],
)
def test_remaining_process_validation(function, arguments):
    with pytest.raises(ValueError):
        function(*arguments)
