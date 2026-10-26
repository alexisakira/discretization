"""Markov-process discretization routines."""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import ncx2

from ._matching import (
    ProcessResult,
    diagnostics,
    relative_weights,
    sequential_match,
    validate_floor,
)
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
    scale = volatility**2 * (-math.expm1(-mean_reversion * step)) / (4 * mean_reversion)
    degrees_freedom = 4 * mean_reversion * long_run_mean / volatility**2
    noncentrality = current_state * decay / scale
    density = ncx2.pdf(future / scale, degrees_freedom, noncentrality) / scale
    # SciPy assigns zero at x=0 even for df=2; use the continuous right limit
    # at Feller equality, consistent with the scaled-Bessel MATLAB formula.
    if degrees_freedom == 2:
        density = np.where(
            future == 0, math.exp(-noncentrality / 2) / (2 * scale), density
        )
    return density


def discrete_cir(
    mean_reversion: float,
    long_run_mean: float,
    volatility: float,
    step: float,
    state_count: int,
    *,
    method: str = "even",
) -> tuple[FloatArray, FloatArray]:
    """CIR on the same automatically selected feasible even grid as MATLAB.

    ``method="even"`` explicitly selects that same corrected construction.
    The historical coverage/exponential grid interface has been retired.
    """
    if method != "even":
        raise ValueError(
            "CIR supports only method='even' with automatic feasible grid selection"
        )
    _validate_cir_model(mean_reversion, long_run_mean, volatility, step)
    if not isinstance(state_count, (int, np.integer)) or state_count < 3:
        raise ValueError("state_count must be an integer of at least three")
    rho = math.exp(-mean_reversion * step)
    c = -math.expm1(-mean_reversion * step)
    eta = volatility**2 / (mean_reversion * long_run_mean)
    n = state_count

    def slacks(r):
        return np.array(
            [
                4 * eta * c * (rho * r + c / 2) - r * r,
                (1 - r) * ((n - rho) * r - c) - eta * (rho * r + c / 2),
                ((rho * n - 1) * r + c) * (n * r - 1) - eta * (rho * n * r + c / 2),
            ]
        )

    coefficients = [
        [-1, 4 * eta * c * rho, 2 * eta * c * c],
        [-(n - rho), n + 1 - 2 * rho - eta * rho, -c * (1 + eta / 2)],
        [n * (rho * n - 1), n + 1 - rho * n * (2 + eta), -c * (1 + eta / 2)],
    ]
    points = [1 / n, 1.0]
    for polynomial in coefficients:
        polynomial = np.asarray(polynomial)
        nonzero = np.flatnonzero(
            np.abs(polynomial) > 100 * np.spacing(max(1.0, np.abs(polynomial).max()))
        )
        if nonzero.size and nonzero[0] < 2:
            for root in np.roots(polynomial[nonzero[0] :]):
                if (
                    abs(root.imag) <= 1e-10 * max(1, abs(root.real))
                    and 1 / n < root.real < 1
                ):
                    points.append(float(root.real))
    merged = []
    for point in sorted(points):
        if not merged or point - merged[-1] > 1e-11 * max(
            1, abs(point), abs(merged[-1])
        ):
            merged.append(point)
    intervals = [
        (a, b)
        for a, b in zip(merged[:-1], merged[1:])
        if (slacks((a + b) / 2) > 0).all()
    ]
    if not intervals:
        raise ValueError("no strictly feasible CIR grid; increase state_count")
    if len(intervals) != 1:
        raise ValueError("disconnected CIR feasible set")
    lower, upper = intervals[0]
    trim = min(1e-4, 1 / n) * (upper - lower)
    r = np.clip(math.log(n) / (n - 1), lower + trim, upper - trim)
    grid = long_run_mean * r * np.arange(1, n + 1)
    transition = np.zeros((n, n))
    scale = volatility**2 * c / (4 * mean_reversion)
    df = 4 * mean_reversion * long_run_mean / volatility**2
    for row, x in enumerate(grid):
        mean = rho * x + c * long_run_mean
        variance = (
            volatility**2 / mean_reversion * c * (rho * x + long_run_mean * c / 2)
        )
        prior = relative_weights(
            ncx2.logpdf(grid / scale, df, x * rho / scale) - math.log(scale)
        )
        z = (grid - mean) / math.sqrt(variance)
        probability, matched = sequential_match(
            grid, np.vstack((z, z * z)), np.array([0.0, 1.0]), prior, 2
        )
        if matched != 2:
            raise RuntimeError(
                "two-moment CIR matching failed on a theoretically feasible grid"
            )
        transition[row] = probability
    return transition, grid


def _ou_grid_and_weights(grid_or_count, stationary_sd):
    if np.ndim(grid_or_count) == 0:
        if not isinstance(grid_or_count, (int, np.integer)) or grid_or_count < 3:
            raise ValueError("state_count must be an integer of at least three")
        grid = np.linspace(
            -stationary_sd * math.sqrt(grid_or_count - 1),
            stationary_sd * math.sqrt(grid_or_count - 1),
            grid_or_count,
        )
    else:
        grid = np.asarray(grid_or_count, dtype=float)
        if grid.ndim != 1:
            raise ValueError("grid must be a one-dimensional vector")
        if grid.size < 3 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
            raise ValueError(
                "grid must contain at least three finite, increasing points"
            )
    weights = np.r_[grid[1] - grid[0], (grid[2:] - grid[:-2]) / 2, grid[-1] - grid[-2]]
    return grid, weights


def _strict_two_moment_grid(grid, rho, variance):
    for mean in rho * grid:
        if not grid[0] < mean < grid[-1]:
            raise ValueError("conditional mean outside the grid")
        left = np.searchsorted(grid, mean, side="right") - 1
        minimum = (mean - grid[left]) * (grid[left + 1] - mean)
        maximum = (mean - grid[0]) * (grid[-1] - mean)
        if not minimum < variance < maximum:
            raise ValueError("conditional variance infeasible on the grid")


def discrete_ou(
    mean_reversion: float, volatility: float, step: float, grid_or_count: ArrayLike
):
    """Centered Gaussian OU, matching exact conditional mean and variance.

    ``grid_or_count`` is an increasing fixed grid or a count N; automatic
    endpoints are +/- stationary_sd*sqrt(N-1). Infeasible grids raise.
    """
    if any(not np.isfinite(v) or v <= 0 for v in (mean_reversion, volatility, step)):
        raise ValueError(
            "mean_reversion, volatility and step must be finite and positive"
        )
    stationary_sd = volatility / math.sqrt(2 * mean_reversion)
    rho = math.exp(-mean_reversion * step)
    variance = stationary_sd**2 * (-math.expm1(-2 * mean_reversion * step))
    grid, weights = _ou_grid_and_weights(grid_or_count, stationary_sd)
    _strict_two_moment_grid(grid, rho, variance)
    transition = np.empty((grid.size, grid.size))
    for row, mean in enumerate(rho * grid):
        z = (grid - mean) / math.sqrt(variance)
        prior = relative_weights(np.log(weights) - 0.5 * z * z)
        probability, matched = sequential_match(
            grid, np.vstack((z, z * z)), np.array([0.0, 1.0]), prior, 2
        )
        if matched != 2:
            raise RuntimeError("two-moment OU matching failed")
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
    *,
    probability_floor: float | None = None,
    return_diagnostics: bool = False,
) -> ProcessResult:
    """Discretize a Gaussian AR(1) process.

    The Python implementation currently supports the evenly spaced grid used
    by the MATLAB reference fixture. Moment counts from one through four use
    the same sequential fallback behavior as MATLAB. Pass
    ``return_diagnostics=True`` for a third output with matched counts and
    standardized errors, or ``probability_floor=1e-8`` for the 0.1.0 algorithm.
    """
    legacy = validate_floor(probability_floor)
    if method != "even":
        raise NotImplementedError("discrete_ar currently supports method='even'")
    if not isinstance(state_count, (int, np.integer)) or state_count < 3:
        raise ValueError("state_count must be an integer of at least three")
    if moment_count not in (1, 2, 3, 4):
        raise ValueError("moment_count must be one of 1, 2, 3, or 4")
    if not np.isfinite(mean):
        raise ValueError("mean must be finite")
    if not np.isfinite(innovation_std) or innovation_std <= 0:
        raise ValueError("innovation_std must be positive")
    if not np.isfinite(persistence) or abs(persistence) >= 1:
        raise ValueError("persistence must have absolute value below one")

    if grid_width is None:
        threshold = 1 - 2 / (state_count - 1)
        if abs(persistence) <= threshold:
            grid_width = math.sqrt(2 * (state_count - 1))
        else:
            grid_width = math.sqrt(state_count - 1)

    if not np.isfinite(grid_width) or grid_width <= 0:
        raise ValueError("grid_width must be finite and positive")
    unconditional_std = innovation_std / math.sqrt(1 - persistence**2)
    grid = np.linspace(
        mean - grid_width * unconditional_std,
        mean + grid_width * unconditional_std,
        state_count,
    )
    transition = np.empty((state_count, state_count))
    scaling = float(np.max(np.abs(grid)))
    target_moments = np.array([0.0, innovation_std**2, 0.0, 3 * innovation_std**4])

    matched = np.zeros(state_count, dtype=int)
    errors = np.zeros((state_count, 4))
    for row, current_state in enumerate(grid):
        conditional_mean = mean * (1 - persistence) + persistence * current_state
        standardized = (grid - conditional_mean) / innovation_std
        prior = np.exp(-0.5 * standardized**2) / (
            innovation_std * math.sqrt(2 * math.pi)
        )
        if not legacy:
            prior = relative_weights(-0.5 * standardized**2)
            powers = standardized[None, :] ** np.arange(1, 5)[:, None]
            probability, matched[row] = sequential_match(
                grid, powers, np.array([0.0, 1.0, 0.0, 3.0]), prior, moment_count
            )
            transition[row] = probability
            errors[row] = powers @ probability - np.array([0.0, 1.0, 0.0, 3.0])
            continue
        prior = np.maximum(prior, probability_floor)

        def moments(order: int):
            powers = np.arange(1, order + 1)[:, None]
            return lambda values: (
                ((values - conditional_mean) / scaling)[None, :] ** powers
            )

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

    if return_diagnostics:
        info = (
            dict(probability_floor=probability_floor, matched_moments=None)
            if legacy
            else diagnostics(moment_count, matched, errors)
        )
        return transition, grid, info
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
    if volatility**2 > 2 * mean_reversion * long_run_mean:
        raise ValueError("CIR parameters must satisfy volatility^2 <= 2*a*b")
