"""Quadrature rules used by the discretization routines."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def gauss_hermite(order: int) -> tuple[FloatArray, FloatArray]:
    """Return physicists' Gauss-Hermite nodes and weights."""
    _validate_order(order)
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes.astype(float), weights.astype(float)


def gauss_legendre(
    order: int, interval: tuple[float, float] = (-1.0, 1.0)
) -> tuple[FloatArray, FloatArray]:
    """Return Gauss-Legendre nodes and weights on a finite interval."""
    _validate_order(order)
    lower, upper = _validate_interval(interval)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half_width = (upper - lower) / 2
    midpoint = (upper + lower) / 2
    return midpoint + half_width * nodes, half_width * weights


def clenshaw_curtis(
    point_count: int, interval: tuple[float, float] = (-1.0, 1.0)
) -> tuple[FloatArray, FloatArray]:
    """Return Clenshaw-Curtis nodes and weights on a finite interval.

    The node order matches MATLAB ``fclencurt``: upper endpoint first.
    """
    _validate_order(point_count)
    lower, upper = _validate_interval(interval)
    if point_count == 1:
        return np.array([(lower + upper) / 2]), np.array([upper - lower])

    degree = point_count - 1
    theta = np.pi * np.arange(point_count) / degree
    canonical_nodes = np.cos(theta)
    weights = np.zeros(point_count)
    interior = np.arange(1, degree)
    accumulator = np.ones(degree - 1)

    if degree % 2 == 0:
        weights[0] = 1 / (degree**2 - 1)
        weights[-1] = weights[0]
        for harmonic in range(1, degree // 2):
            accumulator -= (
                2
                * np.cos(2 * harmonic * theta[interior])
                / (4 * harmonic**2 - 1)
            )
        accumulator -= np.cos(degree * theta[interior]) / (degree**2 - 1)
    else:
        weights[0] = 1 / degree**2
        weights[-1] = weights[0]
        for harmonic in range(1, (degree + 1) // 2):
            accumulator -= (
                2
                * np.cos(2 * harmonic * theta[interior])
                / (4 * harmonic**2 - 1)
            )

    weights[interior] = 2 * accumulator / degree
    half_width = (upper - lower) / 2
    midpoint = (upper + lower) / 2
    return midpoint + half_width * canonical_nodes, half_width * weights


def gaussian_mixture_quadrature(
    coefficients: ArrayLike,
    means: ArrayLike,
    standard_deviations: ArrayLike,
    order: int,
) -> tuple[FloatArray, FloatArray]:
    """Construct Gaussian quadrature for a univariate Gaussian mixture."""
    _validate_order(order)
    coefficient_array = np.asarray(coefficients, dtype=float).reshape(-1)
    mean_array = np.asarray(means, dtype=float).reshape(-1)
    std_array = np.asarray(standard_deviations, dtype=float).reshape(-1)

    if not (
        coefficient_array.size == mean_array.size == std_array.size
        and coefficient_array.size > 0
    ):
        raise ValueError("coefficients, means, and standard_deviations must align")
    if not all(
        np.all(np.isfinite(array))
        for array in (coefficient_array, mean_array, std_array)
    ):
        raise ValueError("mixture parameters must be finite")
    if np.any(coefficient_array < 0) or not np.any(coefficient_array > 0):
        raise ValueError("coefficients must be nonnegative with positive total mass")
    if np.any(std_array <= 0):
        raise ValueError("standard deviations must be positive")

    component_moments = np.zeros((2 * order + 1, coefficient_array.size))
    component_moments[0] = 1
    component_moments[1] = mean_array
    variances = std_array**2
    for moment_order in range(2, 2 * order + 1):
        component_moments[moment_order] = (
            mean_array * component_moments[moment_order - 1]
            + (moment_order - 1)
            * variances
            * component_moments[moment_order - 2]
        )
    moments = component_moments @ coefficient_array
    return _quadrature_from_moments(moments, order, coefficient_array.sum())


def data_driven_gaussian_quadrature(
    data: ArrayLike, order: int
) -> tuple[FloatArray, FloatArray]:
    """Construct Gaussian quadrature directly from one-dimensional data."""
    _validate_order(order)
    observations = np.asarray(data, dtype=float).reshape(-1)
    if observations.size <= 1:
        raise ValueError("sample size must exceed one")
    if not np.all(np.isfinite(observations)):
        raise ValueError("data must be finite")
    if np.all(observations > 0):
        warnings.warn(
            "Positive data may lack sufficiently high moments; consider log(data).",
            RuntimeWarning,
            stacklevel=2,
        )

    mean = float(np.mean(observations))
    standard_deviation = float(np.std(observations, ddof=0))
    if standard_deviation == 0:
        raise ValueError("data must have positive variance")
    standardized = (observations - mean) / standard_deviation
    powers = np.arange(2 * order + 1)[:, None]
    moments = np.mean(standardized[None, :] ** powers, axis=1)
    nodes, weights = _quadrature_from_moments(moments, order, 1.0)
    return mean + standard_deviation * nodes, weights


def _quadrature_from_moments(
    moments: FloatArray, order: int, total_mass: float
) -> tuple[FloatArray, FloatArray]:
    moment_matrix = np.empty((order + 1, order + 1))
    for row in range(order + 1):
        moment_matrix[row] = moments[row : row + order + 1]
    try:
        lower_cholesky = np.linalg.cholesky(moment_matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "moments do not define a positive-definite quadrature problem"
        ) from error

    diagonal = np.diag(lower_cholesky)[:-1]
    beta = diagonal[1:] / diagonal[:-1]
    adjacent = np.diag(lower_cholesky, k=-1)
    ratios = adjacent / diagonal
    alpha = ratios - np.r_[0.0, ratios[:-1]]
    jacobi = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    nodes, eigenvectors = np.linalg.eigh(jacobi)
    weights = total_mass * eigenvectors[0] ** 2
    return nodes, weights


def _validate_order(order: int) -> None:
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")


def _validate_interval(interval: tuple[float, float]) -> tuple[float, float]:
    if len(interval) != 2:
        raise ValueError("interval must contain exactly two endpoints")
    lower, upper = map(float, interval)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("interval must contain finite increasing endpoints")
    return lower, upper
