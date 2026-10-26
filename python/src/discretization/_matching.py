"""Relative log references and sequential, scale-free moment matching."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls
from scipy.special import log_ndtr

from .maxent import entropy_objective, maximum_entropy_weights

ProcessResult = tuple[NDArray, NDArray] | tuple[NDArray, NDArray, dict[str, object]]


def validate_floor(value):
    if value is not None and (
        not np.isscalar(value) or not np.isfinite(value) or value < 0 or value > 1
    ):
        raise ValueError("probability_floor must be None or a finite scalar in [0, 1]")
    return value is not None and value > 0


def relative_weights(log_prior):
    values = np.asarray(log_prior, dtype=float)
    if (
        not np.isfinite(values).any()
        or np.isnan(values).any()
        or np.isposinf(values).any()
    ):
        raise ValueError("reference log weights must have a finite maximum")
    prior = np.exp(np.maximum(values - values.max(), np.log(np.finfo(float).tiny)))
    return prior / prior.sum()


def normal_interval_log_probability(lower, upper):
    positive = lower > 0
    a = np.where(positive, -upper, lower)
    b = np.where(positive, -lower, upper)
    log_a, log_b = log_ndtr(a), log_ndtr(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return log_b + np.log(-np.expm1(log_a - log_b))


def sequential_match(grid, powers, targets, prior, requested, tolerance=1e-7):
    targets = np.asarray(targets)
    probability = np.zeros_like(prior)
    probability[np.argmin(np.abs(powers[0]))] = 1
    if powers[0].min() >= 0 or powers[0].max() <= 0:
        return probability, 0
    matched, dual = 0, np.zeros(0)
    for count in range(1, requested + 1):
        evaluated, target = powers[:count], targets[:count]
        if count == 2:
            z = powers[0]
            minimum_variance = -z[z <= 0].max() * z[z >= 0].min()
            maximum_variance = -z.min() * z.max()
            if targets[1] < minimum_variance or targets[1] > maximum_variance:
                break
        elif count > 2:
            constraints = np.vstack((np.ones(prior.size), evaluated))
            rhs = np.r_[1.0, target]
            _, residual = nnls(constraints, rhs)
            if residual > 1e-10 * (1 + np.linalg.norm(rhs)):
                break
        _, error, _, candidate = entropy_objective(
            np.zeros(count), evaluated, target, prior
        )
        if np.linalg.norm(error, np.inf) <= 1e-12:
            probability, dual, matched = candidate, np.zeros(count), count
            continue
        candidate, candidate_dual, error = maximum_entropy_weights(
            grid,
            lambda unused, evaluated=evaluated: evaluated,
            target,
            prior,
            np.r_[dual, 0.0],
        )
        objective, refined_error, hessian, refined = entropy_objective(
            candidate_dual, evaluated, target, prior
        )
        for _ in range(8):
            if (
                np.linalg.norm(refined_error, np.inf) <= 1e-10
                or np.linalg.cond(hessian) > 1e14
            ):
                break
            step = np.linalg.solve(hessian, refined_error)
            for damping in range(21):
                new_dual = candidate_dual - step * 2.0 ** (-damping)
                if not np.isfinite(new_dual).all():
                    continue
                try:
                    new_objective, new_error, new_hessian, new_probability = (
                        entropy_objective(new_dual, evaluated, target, prior)
                    )
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    continue
                if (
                    np.isfinite(new_objective)
                    and np.linalg.norm(new_error, np.inf)
                    < np.linalg.norm(refined_error, np.inf)
                    and new_objective <= objective + 1e-12 * max(1, abs(objective))
                ):
                    candidate_dual = new_dual
                    objective, refined_error, hessian, refined = (
                        new_objective,
                        new_error,
                        new_hessian,
                        new_probability,
                    )
                    break
            else:
                break
        if np.linalg.norm(refined_error, np.inf) < np.linalg.norm(error, np.inf):
            candidate, error = refined, refined_error
        if (
            not np.isfinite(candidate).all()
            or np.linalg.norm(error, np.inf) > tolerance
        ):
            break
        probability, dual, matched = candidate, candidate_dual, count
    return probability, matched


def diagnostics(requested, matched, errors):
    matched = np.asarray(matched)
    downgraded = matched < requested
    if matched.ndim > 1:
        downgraded = downgraded.any(axis=1)
    return dict(
        requested_moments=requested,
        matched_moments=matched,
        standardized_moment_errors=errors,
        moment_tolerance=1e-7,
        downgraded_rows=np.flatnonzero(downgraded),
        probability_floor=None,
    )
