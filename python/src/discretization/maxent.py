"""Stable maximum-entropy discrete approximation."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


def entropy_objective(
    dual: ArrayLike,
    evaluated_moments: ArrayLike,
    target_moments: ArrayLike,
    prior: ArrayLike,
) -> tuple[float, FloatArray, FloatArray, FloatArray]:
    """Evaluate the log-sum-exp dual objective and its derivatives."""
    dual_array = np.asarray(dual, dtype=float).reshape(-1)
    moments = np.asarray(evaluated_moments, dtype=float)
    targets = np.asarray(target_moments, dtype=float).reshape(-1)
    prior_array = np.asarray(prior, dtype=float).reshape(-1)

    if moments.ndim == 1:
        moments = moments.reshape(1, -1)
    if moments.ndim != 2:
        raise ValueError("evaluated_moments must be a one- or two-dimensional array")

    moment_count, point_count = moments.shape
    if dual_array.size != moment_count or targets.size != moment_count:
        raise ValueError("dual and target_moments must match the moment dimension")
    if prior_array.size != point_count:
        raise ValueError("prior must contain one weight per grid point")
    if not all(
        np.all(np.isfinite(array))
        for array in (dual_array, moments, targets, prior_array)
    ):
        raise ValueError("all inputs must be finite")
    if np.any(prior_array < 0) or not np.any(prior_array > 0):
        raise ValueError("prior must be nonnegative with at least one positive weight")

    differences = moments - targets[:, None]
    log_prior = np.full(point_count, -np.inf)
    positive = prior_array > 0
    log_prior[positive] = np.log(prior_array[positive])
    log_weights = log_prior + dual_array @ differences
    if not np.all(np.isfinite(log_weights[positive])):
        raise FloatingPointError("dual evaluation exceeded floating-point range")
    shift = float(np.max(log_weights))
    scaled_weights = np.exp(log_weights - shift)
    normalizer = float(np.sum(scaled_weights))
    probability = scaled_weights / normalizer

    objective = shift + np.log(normalizer)
    gradient = (differences * probability).sum(axis=1)
    hessian = (differences * probability) @ differences.T
    hessian -= np.outer(gradient, gradient)
    hessian = (hessian + hessian.T) / 2

    return float(objective), gradient, hessian, probability


def maximum_entropy_weights(
    grid: ArrayLike,
    moment_function: Callable[[FloatArray], ArrayLike],
    target_moments: ArrayLike,
    prior: ArrayLike | None = None,
    dual0: ArrayLike | None = None,
    *,
    moment_tolerance: float = 1e-5,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Construct a maximum-entropy distribution on a fixed grid.

    Grid points occupy the last array axis. ``moment_function`` must return
    one row per targeted moment and one column per grid point.
    """
    grid_array = np.asarray(grid, dtype=float)
    if grid_array.ndim == 0:
        raise ValueError("grid must contain at least one dimension")
    point_count = grid_array.shape[-1]

    evaluated = np.asarray(moment_function(grid_array), dtype=float)
    if evaluated.ndim == 1:
        evaluated = evaluated.reshape(1, -1)
    targets = np.asarray(target_moments, dtype=float).reshape(-1)
    if evaluated.ndim != 2 or evaluated.shape[1] != point_count:
        raise ValueError("moment_function returned an incompatible shape")
    if evaluated.shape[0] != targets.size:
        raise ValueError("target_moments does not match the moment function")

    moment_count = targets.size
    if prior is None:
        prior_array = np.full(point_count, 1 / point_count)
    else:
        prior_array = np.asarray(prior, dtype=float).reshape(-1)
    if dual0 is None:
        initial_dual = np.zeros(moment_count)
    else:
        initial_dual = np.asarray(dual0, dtype=float).reshape(-1)

    # Validate dimensions and values before invoking an optimizer.
    entropy_objective(initial_dual, evaluated, targets, prior_array)

    def objective(dual: FloatArray) -> float:
        return entropy_objective(dual, evaluated, targets, prior_array)[0]

    def gradient(dual: FloatArray) -> FloatArray:
        return entropy_objective(dual, evaluated, targets, prior_array)[1]

    def hessian(dual: FloatArray) -> FloatArray:
        return entropy_objective(dual, evaluated, targets, prior_array)[2]

    zero_dual = np.zeros(moment_count)
    _, zero_error, _, _ = entropy_objective(
        zero_dual, evaluated, targets, prior_array
    )
    best_dual = zero_dual
    best_error_norm = float(np.linalg.norm(zero_error))

    attempts: list[tuple[str, FloatArray]] = [("trust-exact", initial_dual)]
    if best_error_norm > moment_tolerance and np.any(initial_dual != 0):
        attempts.append(("trust-exact", zero_dual))
    attempts.append(("BFGS", zero_dual))

    for method, initial in attempts:
        if best_error_norm <= moment_tolerance:
            break
        try:
            kwargs: dict[str, object] = {
                "method": method,
                "jac": gradient,
                "options": {"gtol": 1e-10, "maxiter": 1000},
            }
            if method == "trust-exact":
                kwargs["hess"] = hessian
            with warnings.catch_warnings():
                # Infeasible dual problems can make exploratory optimizer
                # steps overflow before the finite-candidate fallback runs.
                warnings.simplefilter("ignore", RuntimeWarning)
                result = minimize(objective, initial, **kwargs)
            candidate = np.asarray(result.x, dtype=float)
            if np.all(np.isfinite(candidate)):
                _, candidate_error, _, _ = entropy_objective(
                    candidate, evaluated, targets, prior_array
                )
                candidate_error_norm = float(np.linalg.norm(candidate_error))
                if np.isfinite(candidate_error_norm) and candidate_error_norm < best_error_norm:
                    best_dual = candidate
                    best_error_norm = candidate_error_norm
        except (FloatingPointError, ValueError, np.linalg.LinAlgError, OverflowError):
            continue

    _, moment_error, _, probability = entropy_objective(
        best_dual, evaluated, targets, prior_array
    )
    return probability, best_dual, moment_error
