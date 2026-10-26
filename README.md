# discretization
Matlab and Python codes for discretizing non-Gaussian Markov processes
based on "Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments", Quantitative Economics 8(2):651-683 (2017)
https://doi.org/10.3982/QE737
by Leland E. Farmer & Alexis Akira Toda

Please read the [discretization guide](discretization.pdf) for installation,
usage, numerical guidance, compatibility names, and citations.

## Version 0.2.0 numerical change

Default transition matrices change: reference densities no longer receive an
absolute `1e-8` floor. This matters for likelihood-based estimation because
artificial tail mass can accumulate over many observations. Relative log-weight
stabilization retains rare transitions without changing grid-width defaults.

Use MATLAB `'ProbabilityFloor',1e-8` or Python `probability_floor=1e-8` to recover
the 0.1.0 process algorithms. The option floors reference weights before moment
matching; final transition probabilities need not exceed the floor. See
[API.md](API.md) for diagnostics, fallback rules and examples.

Both languages include Gaussian OU and the CIR rewrite with an
automatically selected feasible even grid. All default process routines use
the same moment-matching conventions and conditional-standard-deviation scaling.
New defaults are checked against MATLAB R2026a fixtures; historical R2024b
fixtures remain available for legacy regression tests.

## MATLAB usage

Add only the repository root to the MATLAB path and use the namespaced API:

```matlab
addpath('path/to/discretization-master')
[transition,grid] = discretization.discreteAR(0,0.9,0.1,9);
```

Descriptive names are used for less familiar routines, such as
`discretization.discreteGaussianMixtureAR` and
`discretization.momentMatchedDistribution`. See [API.md](API.md) for the
complete API and the mapping from historical function names.

## Python usage

Install the Python package from the repository root:

```console
python -m pip install ./python
```

The installed package is imported as `discretization`. See
[python/README.md](python/README.md) for examples.

## Modernization branch

The modernization work adds automated MATLAB tests, a numerically stable
log-sum-exp maximum-entropy objective, and a Python port under `python/`.
The Python port includes the maximum-entropy core, Gaussian AR and VAR
processes, Cox-Ingersoll-Ross, AR processes with Gaussian-mixture shocks or
stochastic volatility, nonparametric moment discretization, and reusable
quadrature routines. Model families are checked against deterministic MATLAB reference outputs.
Version 0.2.0 defaults use R2026a fixtures; explicit legacy floors use R2024b fixtures.

The MATLAB implementation has also been profiled and optimized while retaining
pre-optimization numerical reference checks. Reproducible workloads and the
measured R2024b speedups are in [benchmarks](benchmarks/RESULTS.md).
