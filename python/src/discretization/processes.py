"""Multivariate and non-Gaussian Markov-process discretization routines."""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, nnls
from scipy.stats import norm

from .maxent import maximum_entropy_weights
from .quadrature import (
    clenshaw_curtis,
    gauss_hermite,
    gauss_legendre,
    gaussian_mixture_quadrature,
)

FloatArray = NDArray[np.float64]


def discrete_var(
    constant: ArrayLike,
    lag: ArrayLike,
    innovation_covariance: ArrayLike,
    state_count: int,
    moment_count: int = 2,
    method: str = "even",
    grid_width: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Discretize a Gaussian VAR(1) process.

    ``state_count`` is the number of grid points in each dimension. The
    returned state grid has shape ``(dimension, state_count**dimension)``.
    Destination states use tensor-product order with the last dimension
    changing fastest, matching MATLAB ``discretization.discreteVAR``.
    """
    lag_matrix = _square_matrix(lag, "lag")
    dimension = lag_matrix.shape[0]
    constant_vector = np.asarray(constant, dtype=float).reshape(-1)
    covariance = _square_matrix(
        innovation_covariance,
        "innovation_covariance",
    )
    _validate_state_count(state_count)
    _validate_var_moment_count(moment_count)
    normalized_method = _normalize_var_method(method)

    if constant_vector.size != dimension:
        raise ValueError("constant must contain one value per VAR dimension")
    if covariance.shape != lag_matrix.shape:
        raise ValueError("innovation_covariance must have the same shape as lag")
    if not all(
        np.all(np.isfinite(array))
        for array in (constant_vector, lag_matrix, covariance)
    ):
        raise ValueError("VAR parameters must be finite")
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12):
        raise ValueError("innovation_covariance must be symmetric")
    try:
        covariance_cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("innovation_covariance must be positive definite") from error
    if np.max(np.abs(np.linalg.eigvals(lag_matrix))) >= 1:
        raise ValueError("lag must define a stationary VAR process")

    if normalized_method == "quantile":
        warnings.warn(
            "The quantile method is poor and not recommended.",
            RuntimeWarning,
            stacklevel=2,
        )
    if normalized_method == "quadrature" and np.max(
        np.abs(np.linalg.eigvals(lag_matrix))
    ) > 0.8:
        warnings.warn(
            "The quadrature method may perform poorly for persistent processes.",
            RuntimeWarning,
            stacklevel=2,
        )
    if grid_width is None:
        grid_width = math.sqrt(state_count - 1)
    if not np.isfinite(grid_width) or grid_width <= 0:
        raise ValueError("grid_width must be finite and positive")

    gaussian_moments = _standard_normal_moments(moment_count)
    identity = np.eye(dimension)
    unconditional_mean = np.linalg.solve(identity - lag_matrix, constant_vector)

    if dimension == 1:
        transform = covariance_cholesky
        standardized_lag = lag_matrix.copy()
        stationary_covariance = np.array(
            [[1 / (1 - float(lag_matrix[0, 0]) ** 2)]]
        )
    else:
        standardized_lag = np.linalg.solve(
            covariance_cholesky,
            lag_matrix @ covariance_cholesky,
        )
        vectorized_identity = identity.reshape(-1, order="F")
        covariance_vector = np.linalg.solve(
            np.eye(dimension**2)
            - np.kron(standardized_lag, standardized_lag),
            vectorized_identity,
        )
        initial_stationary_covariance = covariance_vector.reshape(
            (dimension, dimension),
            order="F",
        )
        rotation = _minimum_variance_rotation(initial_stationary_covariance)
        standardized_lag = rotation.T @ standardized_lag @ rotation
        stationary_covariance = (
            rotation.T @ initial_stationary_covariance @ rotation
        )
        transform = covariance_cholesky @ rotation

    stationary_covariance = (
        stationary_covariance + stationary_covariance.T
    ) / 2
    stationary_std = np.sqrt(np.diag(stationary_covariance))
    one_dimensional_grids = np.empty((dimension, state_count))
    quantile_bounds: FloatArray | None = None
    quadrature_weights: FloatArray | None = None

    if normalized_method == "even":
        minimum_variance = float(np.min(np.linalg.eigvalsh(stationary_covariance)))
        if minimum_variance <= 0:
            raise ValueError("the stationary covariance must be positive definite")
        endpoint = math.sqrt(minimum_variance) * grid_width
        one_dimensional_grids[:] = np.linspace(-endpoint, endpoint, state_count)
    elif normalized_method == "quantile":
        probabilities = (2 * np.arange(1, state_count + 1) - 1) / (
            2 * state_count
        )
        bound_probabilities = np.arange(1, state_count) / state_count
        quantile_bounds = np.empty((dimension, state_count + 1))
        for index in range(dimension):
            one_dimensional_grids[index] = norm.ppf(
                probabilities,
                scale=stationary_std[index],
            )
            quantile_bounds[index] = np.r_[
                -np.inf,
                norm.ppf(bound_probabilities, scale=stationary_std[index]),
                np.inf,
            ]
    else:
        nodes, quadrature_weights = gauss_hermite(state_count)
        one_dimensional_grids[:] = math.sqrt(2) * nodes

    standardized_states = _cartesian_product_rows(one_dimensional_grids)
    conditional_means = standardized_lag @ standardized_states
    total_state_count = state_count**dimension
    transition = np.zeros((total_state_count, total_state_count))
    scaling_factors = one_dimensional_grids[:, -1]
    previous_dual = np.zeros((dimension, 2))
    minimum_prior = 1e-8

    for row in range(total_state_count):
        if normalized_method == "even":
            prior = norm.pdf(
                one_dimensional_grids,
                loc=conditional_means[:, row, None],
                scale=1,
            )
        elif normalized_method == "quantile":
            assert quantile_bounds is not None
            prior = norm.cdf(
                quantile_bounds[:, 1:],
                loc=conditional_means[:, row, None],
                scale=1,
            ) - norm.cdf(
                quantile_bounds[:, :-1],
                loc=conditional_means[:, row, None],
                scale=1,
            )
        else:
            assert quadrature_weights is not None
            prior = (
                norm.pdf(
                    one_dimensional_grids,
                    loc=conditional_means[:, row, None],
                    scale=1,
                )
                / norm.pdf(one_dimensional_grids, loc=0, scale=1)
                * (quadrature_weights / math.sqrt(math.pi))
            )
        prior = np.maximum(prior, minimum_prior)

        component_probabilities = np.empty((dimension, state_count))
        for component in range(dimension):
            grid = one_dimensional_grids[component]
            conditional_mean = conditional_means[component, row]
            scaling = scaling_factors[component]

            def moments(order: int):
                powers = np.arange(1, order + 1)[:, None]
                return lambda values: (
                    (values - conditional_mean) / scaling
                )[None, :] ** powers

            if moment_count == 1:
                probability, _, _ = maximum_entropy_weights(
                    grid,
                    moments(1),
                    [0.0],
                    prior[component],
                    [0.0],
                )
                component_probabilities[component] = probability
                continue

            targets = gaussian_moments[:2] / scaling ** np.arange(1, 3)
            evaluated_moments = moments(2)(grid)
            if _moment_target_feasible(evaluated_moments, targets):
                probability, dual, moment_error = maximum_entropy_weights(
                    grid,
                    moments(2),
                    targets,
                    prior[component],
                    previous_dual[component],
                )
            else:
                dual = np.zeros(2)
                moment_error = np.array([np.inf])
            if np.linalg.norm(moment_error) > 1e-5:
                warnings.warn(
                    "Failed to match two VAR moments; matching the mean only.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                probability, _, _ = maximum_entropy_weights(
                    grid,
                    moments(1),
                    [0.0],
                    prior[component],
                    [0.0],
                )
                previous_dual[component] = 0
                component_probabilities[component] = probability
                continue

            previous_dual[component] = dual
            current_dual = dual
            for order in range(4, moment_count + 1, 2):
                targets = gaussian_moments[:order] / scaling ** np.arange(
                    1,
                    order + 1,
                )
                candidate, candidate_dual, moment_error = maximum_entropy_weights(
                    grid,
                    moments(order),
                    targets,
                    prior[component],
                    np.r_[current_dual, 0.0, 0.0],
                )
                if np.linalg.norm(moment_error) > 1e-5:
                    warnings.warn(
                        f"Failed to match {order} VAR moments; using {order - 2}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break
                probability = candidate
                current_dual = candidate_dual
            component_probabilities[component] = probability

        row_probability = component_probabilities[0]
        for component in range(1, dimension):
            row_probability = np.kron(
                row_probability,
                component_probabilities[component],
            )
        transition[row] = row_probability

    state_grid = transform @ standardized_states + unconditional_mean[:, None]
    return transition, state_grid


def discrete_gaussian_mixture_ar(
    mean: float,
    ar_coefficients: ArrayLike,
    mixture_probabilities: ArrayLike,
    mixture_means: ArrayLike,
    mixture_standard_deviations: ArrayLike,
    state_count: int,
    moment_count: int = 2,
    method: str = "even",
    grid_width: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Discretize an AR(p) process with Gaussian-mixture innovations."""
    coefficients = np.asarray(ar_coefficients, dtype=float).reshape(-1)
    probabilities = np.asarray(mixture_probabilities, dtype=float).reshape(-1)
    component_means = np.asarray(mixture_means, dtype=float).reshape(-1)
    component_std = np.asarray(
        mixture_standard_deviations,
        dtype=float,
    ).reshape(-1)
    _validate_state_count(state_count)
    if moment_count not in (1, 2, 3, 4):
        raise ValueError("moment_count must be one of 1, 2, 3, or 4")
    normalized_method = _normalize_mixture_method(method)

    if coefficients.size == 0 or not np.all(np.isfinite(coefficients)):
        raise ValueError("ar_coefficients must contain finite values")
    if not np.isfinite(mean):
        raise ValueError("mean must be finite")
    if not (
        probabilities.size
        == component_means.size
        == component_std.size
        and probabilities.size > 0
    ):
        raise ValueError("mixture parameter arrays must have equal positive length")
    if not all(
        np.all(np.isfinite(array))
        for array in (probabilities, component_means, component_std)
    ):
        raise ValueError("mixture parameters must be finite")
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("mixture_probabilities must be nonnegative and sum to one")
    if np.any(component_std <= 0):
        raise ValueError("mixture_standard_deviations must be positive")

    order = coefficients.size
    companion = np.zeros((order, order))
    companion[0] = coefficients
    if order > 1:
        companion[1:, :-1] = np.eye(order - 1)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    if spectral_radius >= 1:
        raise ValueError("ar_coefficients must define a stationary process")

    component_variances = component_std**2
    innovation_moments = np.array(
        [
            probabilities @ component_means,
            probabilities @ (component_means**2 + component_variances),
            probabilities
            @ (component_means**3 + 3 * component_means * component_variances),
            probabilities
            @ (
                component_means**4
                + 6 * component_means**2 * component_variances
                + 3 * component_variances**2
            ),
        ]
    )
    innovation_std = math.sqrt(
        innovation_moments[1] - innovation_moments[0] ** 2
    )
    basis = np.zeros(order**2)
    basis[0] = 1
    inverse_first_column = np.linalg.solve(
        np.eye(order**2) - np.kron(companion, companion),
        basis,
    )
    unconditional_std = innovation_std * math.sqrt(inverse_first_column[0])

    if grid_width is None:
        threshold = 1 - 2 / (state_count - 1)
        if spectral_radius <= threshold:
            grid_width = math.sqrt(2 * (state_count - 1))
        else:
            grid_width = math.sqrt(state_count - 1)
    if not np.isfinite(grid_width) or grid_width <= 0:
        raise ValueError("grid_width must be finite and positive")

    lower = mean - grid_width * unconditional_std
    upper = mean + grid_width * unconditional_std
    if normalized_method == "even":
        one_dimensional_grid = np.linspace(lower, upper, state_count)
        integration_weights = np.ones(state_count)
    elif normalized_method == "gauss_legendre":
        one_dimensional_grid, integration_weights = gauss_legendre(
            state_count,
            (lower, upper),
        )
    elif normalized_method == "clenshaw_curtis":
        nodes, weights = clenshaw_curtis(state_count, (lower, upper))
        one_dimensional_grid = nodes[::-1]
        integration_weights = weights[::-1]
    elif normalized_method == "gauss_hermite":
        if spectral_radius > 0.8:
            warnings.warn(
                "The model is persistent; the even grid is recommended.",
                RuntimeWarning,
                stacklevel=2,
            )
        nodes, weights = gauss_hermite(state_count)
        one_dimensional_grid = mean + math.sqrt(2) * innovation_std * nodes
        integration_weights = weights / math.sqrt(math.pi)
    else:
        if spectral_radius > 0.8:
            warnings.warn(
                "The model is persistent; the even grid is recommended.",
                RuntimeWarning,
                stacklevel=2,
            )
        nodes, integration_weights = gaussian_mixture_quadrature(
            probabilities,
            component_means,
            component_std,
            state_count,
        )
        one_dimensional_grid = nodes + mean

    def mixture_pdf(values: ArrayLike) -> FloatArray:
        value_array = np.asarray(values, dtype=float).reshape(-1, 1)
        return np.sum(
            norm.pdf(
                value_array,
                loc=component_means,
                scale=component_std,
            )
            * probabilities,
            axis=1,
        )

    state_grid = _cartesian_product_rows(
        np.tile(one_dimensional_grid, (order, 1))
    )
    total_state_count = state_count**order
    lag_state_count = state_count ** (order - 1)
    transition = np.zeros((total_state_count, total_state_count))
    scaling = float(np.max(np.abs(one_dimensional_grid)))
    minimum_prior = 1e-8
    reference_pdf: FloatArray | None = None
    if normalized_method == "gauss_hermite":
        reference_pdf = norm.pdf(
            one_dimensional_grid,
            loc=0,
            scale=innovation_std,
        )
    elif normalized_method == "gaussian_mixture":
        reference_pdf = mixture_pdf(one_dimensional_grid)

    for row in range(total_state_count):
        conditional_mean = mean * (1 - coefficients.sum()) + coefficients @ state_grid[:, row]
        centered_grid = one_dimensional_grid - conditional_mean
        if reference_pdf is None:
            prior = integration_weights * mixture_pdf(centered_grid)
        else:
            prior = (
                integration_weights
                * mixture_pdf(centered_grid)
                / reference_pdf
            )
        prior = np.maximum(prior, minimum_prior)
        standardized = centered_grid / scaling
        powers = standardized[None, :] ** np.arange(1, 5)[:, None]
        scaled_targets = innovation_moments / scaling ** np.arange(1, 5)

        if moment_count == 1:
            probability, _, _ = maximum_entropy_weights(
                one_dimensional_grid,
                lambda values: powers[:1],
                scaled_targets[:1],
                prior,
                [0.0],
            )
        else:
            if _moment_target_feasible(powers[:2], scaled_targets[:2]):
                probability, dual, moment_error = maximum_entropy_weights(
                    one_dimensional_grid,
                    lambda values: powers[:2],
                    scaled_targets[:2],
                    prior,
                    np.zeros(2),
                )
            else:
                dual = np.zeros(2)
                moment_error = np.array([np.inf])
            if np.linalg.norm(moment_error) > 1e-5:
                warnings.warn(
                    "Failed to match two mixture-AR moments; matching one.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                probability, _, _ = maximum_entropy_weights(
                    one_dimensional_grid,
                    lambda values: powers[:1],
                    scaled_targets[:1],
                    prior,
                    [0.0],
                )
            elif moment_count > 2:
                matched_higher_moment = False
                if moment_count == 4 and _moment_target_feasible(
                    powers,
                    scaled_targets,
                ):
                    candidate, _, moment_error = maximum_entropy_weights(
                        one_dimensional_grid,
                        lambda values: powers,
                        scaled_targets,
                        prior,
                        np.r_[dual, 0.0, 0.0],
                    )
                    if np.linalg.norm(moment_error) <= 1e-5:
                        probability = candidate
                        matched_higher_moment = True

                if not matched_higher_moment:
                    if _moment_target_feasible(powers[:3], scaled_targets[:3]):
                        candidate, _, moment_error = maximum_entropy_weights(
                            one_dimensional_grid,
                            lambda values: powers[:3],
                            scaled_targets[:3],
                            prior,
                            np.r_[dual, 0.0],
                        )
                    else:
                        moment_error = np.array([np.inf])
                    if np.linalg.norm(moment_error) > 1e-5:
                        warnings.warn(
                            "Failed to match three mixture-AR moments; using two.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        probability = candidate
                        if moment_count == 4:
                            warnings.warn(
                                "Failed to match four mixture-AR moments; using three.",
                                RuntimeWarning,
                                stacklevel=2,
                            )

        lag_index = row // state_count
        destinations = np.arange(
            lag_index,
            total_state_count,
            lag_state_count,
        )
        transition[row, destinations] = probability

    return transition, state_grid


def discrete_stochastic_volatility_ar(
    persistence: float,
    volatility_persistence: float,
    unconditional_innovation_std: float,
    volatility_innovation_std: float,
    state_count: int,
    volatility_state_count: int,
    method: str = "even",
    grid_width: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Discretize an AR(1) process with log AR(1) stochastic volatility."""
    _validate_state_count(state_count)
    _validate_state_count(volatility_state_count)
    if not all(
        np.isfinite(value)
        for value in (
            persistence,
            volatility_persistence,
            unconditional_innovation_std,
            volatility_innovation_std,
        )
    ):
        raise ValueError("stochastic-volatility parameters must be finite")
    if abs(persistence) >= 1 or abs(volatility_persistence) >= 1:
        raise ValueError("both persistence parameters must have magnitude below one")
    if unconditional_innovation_std <= 0 or volatility_innovation_std <= 0:
        raise ValueError("both innovation standard deviations must be positive")
    normalized_method = _normalize_var_method(method)
    if grid_width is None:
        grid_width = math.sqrt((state_count - 1) / 2)
    if not np.isfinite(grid_width) or grid_width <= 0:
        raise ValueError("grid_width must be finite and positive")

    unconditional_log_variance = volatility_innovation_std**2 / (
        1 - volatility_persistence**2
    )
    log_variance_mean = (
        2 * math.log(unconditional_innovation_std)
        - unconditional_log_variance / 2
    )
    unconditional_state_std = math.sqrt(
        math.exp(log_variance_mean + unconditional_log_variance / 2)
        / (1 - persistence**2)
    )

    volatility_transition, volatility_grid_matrix = discrete_var(
        [log_variance_mean * (1 - volatility_persistence)],
        [[volatility_persistence]],
        [[volatility_innovation_std**2]],
        volatility_state_count,
        moment_count=2,
        method=normalized_method,
    )
    volatility_grid = volatility_grid_matrix.reshape(-1)
    state_grid_1d = np.linspace(
        -grid_width * unconditional_state_std,
        grid_width * unconditional_state_std,
        state_count,
    )
    total_state_count = state_count * volatility_state_count
    joint_grid = np.vstack(
        (
            np.repeat(state_grid_1d, volatility_state_count),
            np.tile(volatility_grid, state_count),
        )
    )
    transition = np.zeros((total_state_count, total_state_count))
    scaling = float(np.max(np.abs(state_grid_1d)))
    minimum_prior = 1e-8

    for row in range(total_state_count):
        conditional_mean = persistence * joint_grid[0, row]
        conditional_variance = math.exp(
            (1 - volatility_persistence) * log_variance_mean
            + volatility_persistence * joint_grid[1, row]
            + volatility_innovation_std**2 / 2
        )
        prior = norm.pdf(
            state_grid_1d,
            loc=conditional_mean,
            scale=math.sqrt(conditional_variance),
        )
        prior = np.maximum(prior, minimum_prior)

        def moments(values: FloatArray) -> FloatArray:
            centered = (values - conditional_mean) / scaling
            return np.vstack((centered, centered**2))

        targets = np.array([0.0, conditional_variance / scaling**2])
        evaluated_moments = moments(state_grid_1d)
        if _moment_target_feasible(evaluated_moments, targets):
            probability, _, moment_error = maximum_entropy_weights(
                state_grid_1d,
                moments,
                targets,
                prior,
                np.zeros(2),
            )
        else:
            moment_error = np.array([np.inf])
        if np.linalg.norm(moment_error) > 1e-5:
            warnings.warn(
                "Failed to match two stochastic-volatility moments; matching one.",
                RuntimeWarning,
                stacklevel=2,
            )
            probability, _, _ = maximum_entropy_weights(
                state_grid_1d,
                lambda values: (
                    (values - conditional_mean) / scaling
                )[None, :],
                [0.0],
                prior,
                [0.0],
            )
        transition[row] = np.kron(
            probability,
            volatility_transition[row % volatility_state_count],
        )

    return transition, joint_grid


def _square_matrix(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return array


def _validate_state_count(state_count: int) -> None:
    if not isinstance(state_count, (int, np.integer)) or state_count < 3:
        raise ValueError("state_count must be an integer of at least three")


def _validate_var_moment_count(moment_count: int) -> None:
    if not isinstance(moment_count, (int, np.integer)) or (
        moment_count != 1 and (moment_count < 2 or moment_count % 2 != 0)
    ):
        raise ValueError("moment_count must be one or a positive even integer")


def _normalize_var_method(method: str) -> str:
    if not isinstance(method, str):
        raise ValueError("method must be a string")
    normalized = method.lower().replace("-", "_")
    aliases = {
        "even": "even",
        "quantile": "quantile",
        "quadrature": "quadrature",
        "gauss_hermite": "quadrature",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(
            "method must be 'even', 'quantile', or 'quadrature'"
        ) from error


def _normalize_mixture_method(method: str) -> str:
    if not isinstance(method, str):
        raise ValueError("method must be a string")
    normalized = method.lower().replace("-", "_")
    aliases = {
        "even": "even",
        "gauss_legendre": "gauss_legendre",
        "clenshaw_curtis": "clenshaw_curtis",
        "gauss_hermite": "gauss_hermite",
        "gmq": "gaussian_mixture",
        "gaussian_mixture": "gaussian_mixture",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(
            "method must be even, gauss_legendre, clenshaw_curtis, "
            "gauss_hermite, or gaussian_mixture"
        ) from error


def _standard_normal_moments(moment_count: int) -> FloatArray:
    moments = np.zeros(moment_count)
    value = 1.0
    for order in range(2, moment_count + 1, 2):
        value *= order - 1
        moments[order - 1] = value
    return moments


def _cartesian_product_rows(rows: FloatArray) -> FloatArray:
    meshes = np.meshgrid(*rows, indexing="ij")
    return np.vstack([mesh.reshape(-1) for mesh in meshes])


def _minimum_variance_rotation(covariance: FloatArray) -> FloatArray:
    dimension = covariance.shape[0]
    if dimension == 1:
        return np.ones((1, 1))
    if dimension == 2:
        off_diagonal = (covariance[0, 1] + covariance[1, 0]) / 2
        angle = 0.5 * math.atan2(
            covariance[1, 1] - covariance[0, 0],
            2 * off_diagonal,
        )
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return np.array([[cosine, -sine], [sine, cosine]])

    pairs = [
        (row, column)
        for row in range(dimension - 1)
        for column in range(row + 1, dimension)
    ]
    target_diagonal = np.trace(covariance) / dimension

    def rotation_from_angles(angles: FloatArray) -> FloatArray:
        rotation = np.eye(dimension)
        for angle, (row, column) in zip(angles, pairs, strict=True):
            givens = np.eye(dimension)
            cosine = math.cos(angle)
            sine = math.sin(angle)
            givens[row, row] = cosine
            givens[column, column] = cosine
            givens[row, column] = -sine
            givens[column, row] = sine
            rotation = rotation @ givens
        return rotation

    def objective(angles: FloatArray) -> float:
        rotation = rotation_from_angles(angles)
        diagonal = np.diag(rotation.T @ covariance @ rotation)
        return float(np.linalg.norm(diagonal - target_diagonal) ** 2)

    result = minimize(
        objective,
        np.zeros(len(pairs)),
        method="BFGS",
        options={"gtol": 1e-12, "maxiter": 2000},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("failed to construct the VAR variance rotation")
    rotation = rotation_from_angles(np.asarray(result.x, dtype=float))
    diagonal_error = np.linalg.norm(
        np.diag(rotation.T @ covariance @ rotation) - target_diagonal
    )
    if diagonal_error > 1e-6 * (1 + np.linalg.norm(covariance)):
        raise RuntimeError("failed to equalize the rotated VAR variances")
    return rotation


def _moment_target_feasible(
    evaluated_moments: FloatArray,
    target_moments: FloatArray,
    tolerance: float = 1e-10,
) -> bool:
    constraint_matrix = np.vstack(
        (np.ones(evaluated_moments.shape[1]), evaluated_moments)
    )
    constraint_target = np.r_[1.0, target_moments]
    _, residual = nnls(constraint_matrix, constraint_target)
    return bool(residual <= tolerance * (1 + np.linalg.norm(constraint_target)))
