# Farmer–Toda discretization for Python

This package is the Python port of the MATLAB routines in this repository. It
provides the maximum-entropy core, Gaussian AR and VAR processes,
Cox-Ingersoll-Ross, AR processes with Gaussian-mixture shocks or stochastic
volatility, nonparametric moment discretization, and reusable quadrature
routines. Numerical parity is tested against MATLAB R2024b.

## Installation

```console
python -m pip install farmer-toda-discretization
```

From a source checkout, run `python -m pip install .` inside the `python`
directory instead.

Python 3.10 or newer, NumPy, and SciPy are required.

```python
from discretization import discrete_ar

transition, grid = discrete_ar(
    mean=0.0,
    persistence=0.9,
    innovation_std=0.1,
    state_count=9,
)
```

```python
from discretization import discrete_cir, moment_matched_distribution

cir_transition, cir_grid = discrete_cir(
    mean_reversion=0.5,
    long_run_mean=0.03,
    volatility=0.1,
    step=0.25,
    state_count=9,
)

np_grid, probability = moment_matched_distribution(9, [0, 1, 0, 3])
```

```python
import numpy as np

from discretization import (
    discrete_gaussian_mixture_ar,
    discrete_stochastic_volatility_ar,
    discrete_var,
)

var_transition, var_grid = discrete_var(
    constant=np.zeros(2),
    lag=[[0.9, 0.0], [0.0, 0.8]],
    innovation_covariance=[[0.01, 0.0], [0.0, 0.02]],
    state_count=5,
)

mixture_transition, mixture_grid = (
    discrete_gaussian_mixture_ar(
        mean=0.0,
        ar_coefficients=[0.6],
        mixture_probabilities=[0.2, 0.8],
        mixture_means=[-0.1, 0.025],
        mixture_standard_deviations=[0.2, 0.05],
        state_count=7,
    )
)

sv_transition, sv_grid = discrete_stochastic_volatility_ar(
    persistence=0.95,
    volatility_persistence=0.9,
    unconditional_innovation_std=0.007,
    volatility_innovation_std=0.06,
    state_count=9,
    volatility_state_count=5,
)
```

Python returns row-stochastic transition matrices. Multivariate and joint-state
grids have one row per state variable and one column per Markov state, matching
MATLAB's tensor-product ordering.

Python names follow the namespaced MATLAB API in snake case. See the
[API guide](https://github.com/alexisakira/discretization/blob/master/API.md)
for the complete cross-language mapping.

## Citation

When using this package in research, please cite:

> Farmer, L. E., and A. A. Toda (2017). “Discretizing Nonlinear,
> Non-Gaussian Markov Processes with Exact Conditional Moments.”
> *Quantitative Economics* 8(2), 651–683.
> <https://doi.org/10.3982/QE737>

The repository's
[user guide](https://github.com/alexisakira/discretization/blob/master/discretization.pdf)
documents the full API, numerical guidance, and additional references.
