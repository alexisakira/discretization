# Public API and naming

Add the repository root to the MATLAB path and call functions through the
`discretization` package:

```matlab
addpath('path/to/discretization-master')
[transition,grid] = discretization.discreteAR(0,0.9,0.1,9);
```

## Canonical MATLAB names

| Purpose | Canonical function |
| --- | --- |
| Gaussian AR(1) | `discretization.discreteAR` |
| Gaussian VAR(1) | `discretization.discreteVAR` |
| Cox-Ingersoll-Ross process | `discretization.discreteCIR` |
| AR with Gaussian-mixture shocks | `discretization.discreteGaussianMixtureAR` |
| AR with stochastic volatility | `discretization.discreteStochasticVolatilityAR` |
| Distribution from centered moments | `discretization.momentMatchedDistribution` |
| Maximum-entropy weights on a fixed grid | `discretization.maximumEntropyWeights` |
| Gaussian-mixture quadrature | `discretization.gaussianMixtureQuadrature` |
| Data-driven Gaussian quadrature | `discretization.dataDrivenGaussianQuadrature` |
| CIR transition density | `discretization.cirTransitionDensity` |

The package contains its own private numerical helpers. Users therefore need
only the repository root on the MATLAB path, rather than every source folder.

## Compatibility names

Historical entry points remain available for existing scripts. They forward to
the package API and issue migration warnings.

| Historical call | Replacement |
| --- | --- |
| `discreteAR` | `discretization.discreteAR` |
| `discreteVAR` | `discretization.discreteVAR` |
| `discreteCIR` | `discretization.discreteCIR` |
| `discreteGMAR` | `discretization.discreteGaussianMixtureAR` |
| `discreteSV` | `discretization.discreteStochasticVolatilityAR` |
| `discreteNP` | `discretization.momentMatchedDistribution` |
| `discreteApproximation` | `discretization.maximumEntropyWeights` |
| `GaussianMixtureQuadrature` | `discretization.gaussianMixtureQuadrature` |
| `NPGQ` | `discretization.dataDrivenGaussianQuadrature` |
| `CIRpdf` | `discretization.cirTransitionDensity` |

The earlier package names
`discretization.discreteARWithGaussianMixtureShocks`,
`discretization.discreteARWithStochasticVolatility`,
`discretization.discreteDistributionFromMoments`, and
`discretization.nonparametricGaussianQuadrature` are also compatibility
wrappers for their canonical replacements.

Compatibility entry points will remain during the modernization period. Their
removal, if any, should occur only in a future major release.

## Gaussian-mixture AR on an evenly spaced grid

The optional third output reports accuracy by transition row:

```matlab
[P,X,info] = discretization.discreteGaussianMixtureAR( ...
    0,0.95,[0.05 0.95],[-0.8 0.8*0.05/0.95],[0.2 0.03], ...
    127,4,'even',4);
```

The final argument fixes the grid half-width in stationary standard
deviations. Choose width and resolution separately when checking convergence;
the historical default width expands with the point count. A narrow mixture
component can require substantially more points than a Gaussian innovation.

`info.matchedMoments` gives the number of moments matched in each row;
zero means even the conditional mean could not be matched.
`info.downgradedRows` identifies fallback rows. `info.standardizedMomentErrors`
has four columns containing centered conditional-moment errors in powers of
the innovation standard deviation, with acceptance tolerance
`info.momentTolerance`. Diagnostics describe moment accuracy, not a bound on
tail-probability error. Check tails separately when they matter.

The AR location `mu` is the unconditional mean only when the weighted
innovation mean is zero. Otherwise the unconditional mean is
`mu + sum(pC.*muC)/(1-sum(A))`. Component means need not individually be zero.
The optional quadrature methods retain their historical implementation and
return empty diagnostics.

## Python names

Python uses the same concepts in snake case:

```python
from discretization import (
    discrete_ar,
    discrete_cir,
    discrete_gaussian_mixture_ar,
    discrete_stochastic_volatility_ar,
    discrete_var,
    maximum_entropy_weights,
    moment_matched_distribution,
)
```

| MATLAB concept | Python function |
| --- | --- |
| `discreteAR` | `discrete_ar` |
| `discreteVAR` | `discrete_var` |
| `discreteCIR` | `discrete_cir` |
| `discreteGaussianMixtureAR` | `discrete_gaussian_mixture_ar` |
| `discreteStochasticVolatilityAR` | `discrete_stochastic_volatility_ar` |
| `momentMatchedDistribution` | `moment_matched_distribution` |
| `maximumEntropyWeights` | `maximum_entropy_weights` |
| `gaussianMixtureQuadrature` | `gaussian_mixture_quadrature` |
| `dataDrivenGaussianQuadrature` | `data_driven_gaussian_quadrature` |
| `cirTransitionDensity` | `cir_transition_density` |
