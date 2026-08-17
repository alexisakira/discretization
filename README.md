# discretization
Matlab codes for discretizing non-Gaussian Markov processes
based on "Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments", Quantitative Economics 8(2):651-683 (2017)
https://doi.org/10.3982/QE737
by Leland E. Farmer & Alexis Akira Toda

Please read the [discretization guide](discretization.pdf) for installation,
usage, numerical guidance, compatibility names, and citations.

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
quadrature routines. Each model family is checked against deterministic MATLAB
R2024b reference outputs.

The MATLAB implementation has also been profiled and optimized while retaining
pre-optimization numerical reference checks. Reproducible workloads and the
measured R2024b speedups are in [benchmarks](benchmarks/RESULTS.md).
