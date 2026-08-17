"""Discretization of nonparametric distributions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .maxent import maximum_entropy_weights

FloatArray = NDArray[np.float64]


def moment_matched_distribution(
    state_count: int,
    centered_moments: ArrayLike,
    prior: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Discretize a distribution specified by its centered moments."""
    if not isinstance(state_count, (int, np.integer)) or state_count < 2:
        raise ValueError("state_count must be an integer of at least two")
    moments = np.asarray(centered_moments, dtype=float).reshape(-1)
    if moments.size < 2:
        raise ValueError("at least two centered moments are required")
    if moments.size > state_count + 1:
        raise ValueError("there are not enough grid points to match the moments")
    if not np.all(np.isfinite(moments)):
        raise ValueError("centered_moments must be finite")
    if moments[0] != 0:
        raise ValueError("the first centered moment must be zero")
    if moments[1] <= 0:
        raise ValueError("the second centered moment must be positive")

    standard_deviation = math.sqrt(float(moments[1]))
    grid = np.linspace(
        -standard_deviation * math.sqrt(2 * state_count),
        standard_deviation * math.sqrt(2 * state_count),
        state_count,
    )
    if prior is None:
        prior_array = np.exp(-0.5 * (grid / standard_deviation) ** 2)
        prior_array /= prior_array.sum()
    else:
        prior_array = np.asarray(prior, dtype=float).reshape(-1)
        if prior_array.size != state_count:
            raise ValueError("prior must contain one value per grid point")

    powers = np.arange(1, moments.size + 1)[:, None]
    probability, _, _ = maximum_entropy_weights(
        grid,
        lambda values: values[None, :] ** powers,
        moments,
        prior_array,
    )
    return grid, probability
