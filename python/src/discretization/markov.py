"""Markov-process discretization routines."""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import gamma, ncx2

from .maxent import maximum_entropy_weights

FloatArray = NDArray[np.float64]


def cir_transition_density(
    future_state: ArrayLike,
    current_state: float,
    mean_reversion: float,
    long_run_mean: float,
    volatility: float,
    step: float,
) -> FloatArray:
    """Evaluate the Cox-Ingersoll-Ross transition density."""
    future = np.asarray(future_state, dtype=float)
    _validate_cir_model(mean_reversion, long_run_mean, volatility, step)
    if not np.isfinite(current_state) or current_state <= 0:
        raise ValueError("current_state must be finite and positive")
    if np.any(~np.isfinite(future)) or np.any(future < 0):
        raise ValueError("future_state must be finite and nonnegative")

    decay = math.exp(-mean_reversion * step)
    scale = volatility**2 * (1 - decay) / (4 * mean_reversion)
    degrees_freedom = 4 * mean_reversion * long_run_mean / volatility**2
    noncentrality = current_state * decay / scale
    return ncx2.pdf(future / scale, degrees_freedom, noncentrality) / scale


def discrete_cir(
    mean_reversion: float,
    long_run_mean: float,
    volatility: float,
    step: float,
    state_count: int,
    coverage: float = 0.999,
    method: str = "exponential",
) -> tuple[FloatArray, FloatArray]:
    """Discretize a Cox-Ingersoll-Ross process."""
    _validate_cir_model(mean_reversion, long_run_mean, volatility, step)
    if not isinstance(state_count, (int, np.integer)) or state_count < 2:
        raise ValueError("state_count must be an integer of at least two")
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one")
    if method not in {"even", "exponential"}:
        raise ValueError("method must be 'even' or 'exponential'")

    shape = 2 * mean_reversion * long_run_mean / volatility**2
    rate = 2 * mean_reversion / volatility**2
    tail = (1 - coverage) / 2
    endpoints = gamma.ppf([tail, 1 - tail], a=shape, scale=1 / rate)
    if method == "even":
        grid = np.linspace(endpoints[0], endpoints[1], state_count)
        integration_weights = np.ones(state_count)
    else:
        grid = np.geomspace(endpoints[0], endpoints[1], state_count)
        integration_weights = grid.copy()
    integration_weights /= integration_weights.sum()

    transition = np.empty((state_count, state_count))
    scaling = float(np.max(np.abs(grid)))
    decay = math.exp(-mean_reversion * step)

    for row, current_state in enumerate(grid):
        conditional_mean = (
            current_state * decay + long_run_mean * (1 - decay)
        )
        conditional_variance = (
            volatility**2
            / mean_reversion
            * (1 - decay)
            * (current_state * decay + long_run_mean / 2)
        )
        prior = integration_weights * cir_transition_density(
            grid,
            float(current_state),
            mean_reversion,
            long_run_mean,
            volatility,
            step,
        )

        def two_moments(values: FloatArray) -> FloatArray:
            centered = (values - conditional_mean) / scaling
            return np.vstack((centered, centered**2))

        probability, _, moment_error = maximum_entropy_weights(
            grid,
            two_moments,
            [0.0, conditional_variance / scaling**2],
            prior,
            np.zeros(2),
        )
        if np.linalg.norm(moment_error) > 1e-5:
            warnings.warn(
                "Failed to match two CIR moments; matching the mean only.",
                RuntimeWarning,
                stacklevel=2,
            )
            probability, _, _ = maximum_entropy_weights(
                grid,
                lambda values: (values - conditional_mean)[None, :] / scaling,
                [0.0],
                prior,
                [0.0],
            )
        transition[row] = probability

    return transition, grid


def discrete_ar(
    mean: float,
    persistence: float,
    innovation_std: float,
    state_count: int,
    method: str = "even",
    moment_count: int = 2,
    grid_width: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Discretize a Gaussian AR(1) process.

    The Python implementation currently supports the evenly spaced grid used
    by the MATLAB reference fixture. Moment counts from one through four use
    the same sequential fallback behavior as MATLAB.
    """
    if method != "even":
        raise NotImplementedError("discrete_ar currently supports method='even'")
    if not isinstance(state_count, (int, np.integer)) or state_count < 3:
        raise ValueError("state_count must be an integer of at least three")
    if moment_count not in (1, 2, 3, 4):
        raise ValueError("moment_count must be one of 1, 2, 3, or 4")
    if innovation_std <= 0:
        raise ValueError("innovation_std must be positive")
    if abs(persistence) >= 1:
        raise ValueError("persistence must have absolute value below one")

    if grid_width is None:
        threshold = 1 - 2 / (state_count - 1)
        if abs(persistence) <= threshold:
            grid_width = math.sqrt(2 * (state_count - 1))
        else:
            grid_width = math.sqrt(state_count - 1)

    unconditional_std = innovation_std / math.sqrt(1 - persistence**2)
    grid = np.linspace(
        mean - grid_width * unconditional_std,
        mean + grid_width * unconditional_std,
        state_count,
    )
    transition = np.empty((state_count, state_count))
    scaling = float(np.max(np.abs(grid)))
    target_moments = np.array(
        [0.0, innovation_std**2, 0.0, 3 * innovation_std**4]
    )

    for row, current_state in enumerate(grid):
        conditional_mean = mean * (1 - persistence) + persistence * current_state
        standardized = (grid - conditional_mean) / innovation_std
        prior = np.exp(-0.5 * standardized**2) / (
            innovation_std * math.sqrt(2 * math.pi)
        )
        prior = np.maximum(prior, 1e-8)

        def moments(order: int):
            powers = np.arange(1, order + 1)[:, None]
            return lambda values: (
                (values - conditional_mean) / scaling
            )[None, :] ** powers

        if moment_count == 1:
            probability, _, _ = maximum_entropy_weights(
                grid, moments(1), [0.0], prior, [0.0]
            )
            transition[row] = probability
            continue

        probability, dual, error = maximum_entropy_weights(
            grid,
            moments(2),
            target_moments[:2] / scaling ** np.arange(1, 3),
            prior,
            np.zeros(2),
        )
        if np.linalg.norm(error) > 1e-5:
            probability, _, _ = maximum_entropy_weights(
                grid, moments(1), [0.0], prior, [0.0]
            )
            transition[row] = probability
            continue

        if moment_count >= 3:
            candidate, _, error = maximum_entropy_weights(
                grid,
                moments(3),
                target_moments[:3] / scaling ** np.arange(1, 4),
                prior,
                np.r_[dual, 0.0],
            )
            if np.linalg.norm(error) <= 1e-5:
                probability = candidate
            elif moment_count == 3:
                warnings.warn(
                    "Failed to match three moments; using two.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if moment_count == 4:
            candidate, _, error = maximum_entropy_weights(
                grid,
                moments(4),
                target_moments / scaling ** np.arange(1, 5),
                prior,
                np.r_[dual, 0.0, 0.0],
            )
            if np.linalg.norm(error) <= 1e-5:
                probability = candidate
            else:
                warnings.warn(
                    "Failed to match four moments; using a lower-order match.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        transition[row] = probability

    return transition, grid

def _validate_cir_model(
    mean_reversion: float,
    long_run_mean: float,
    volatility: float,
    step: float,
) -> None:
    parameters = np.array(
        [mean_reversion, long_run_mean, volatility, step],
        dtype=float,
    )
    if np.any(~np.isfinite(parameters)) or np.any(parameters <= 0):
        raise ValueError("CIR parameters must be finite and positive")
    if volatility**2 >= 2 * mean_reversion * long_run_mean:
        raise ValueError("CIR parameters must satisfy volatility^2 < 2*a*b")
