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

## Reference weights, fallback rules, and legacy reproduction

MATLAB `discreteAR`, `discreteVAR`, `discreteGaussianMixtureAR`, and
`discreteStochasticVolatilityAR` accept the name-value option `ProbabilityFloor`
(default `0`) after any optional positional arguments. Empty positional arguments
use their defaults. Python counterparts accept the keyword `probability_floor`
(default `None`; `0` is equivalent). Values must be finite scalars in `[0,1]`.

```matlab
[P,X,d] = discretization.discreteAR(0,0.9,0.1,81);
[Pold,Xold] = discretization.discreteAR(0,0.9,0.1,81, ...
    'ProbabilityFloor',1e-8);
```

```python
P, grid, info = discrete_ar(0, 0.9, 0.1, 81, return_diagnostics=True)
Pold, old_grid = discrete_ar(0, 0.9, 0.1, 81, probability_floor=1e-8)
```

Default reference weights are computed in log space, shifted by their row maximum,
clamped at `log(realmin)`, exponentiated and normalized. No absolute density floor
is used. Explicit positive floors select the 0.1.0 algorithm, preserving its
unscaled density floor and solver sequence; `1e-8` recovers that release within
solver/platform precision. The compatibility option covers routines that had
the old floor; it does not revert the CIR rewrite or add a floor to OU.

For defaults, moments are expressed in conditional-innovation SD units. The
matcher attempts the mean, then variance, then higher moments sequentially,
retaining the last accepted distribution. For centered standardized nodes `z`,
strict mean feasibility requires `min(z)<0<max(z)`; variance feasibility additionally
requires `-min(z)*max(z)>=1`. A convex-hull check also catches the minimum variance
implied by grid spacing and infeasible higher moments. If the mean is infeasible
or its numerical solve fails, use a point mass on the nearest node and report
zero matched moments. Higher-moment failure retains the lower-order match.

The third MATLAB output and Python `return_diagnostics=True` expose
`matchedMoments` / `matched_moments`, `requestedMoments` / `requested_moments`,
`standardizedMomentErrors` / `standardized_moment_errors`, `momentTolerance` /
`moment_tolerance` (`1e-7`), and `downgradedRows` / `downgraded_rows`. VAR counts
and residuals are per rotated dimension. SV also constructs its volatility chain
with the same floor convention. MATLAB row indices are one-based; Python indices
are zero-based. Explicit legacy-floor runs return empty/`None` matched counts.

Moment projection does not necessarily preserve the normalized conditional
density near a truncated boundary. Check grid width and resolution separately,
especially in likelihood calculations. A reference floor is not a lower bound
on the probabilities after projection. User-supplied priors in
`maximumEntropyWeights` / `maximum_entropy_weights` retain their exact support;
those core routines do not inject floors.

## CIR and OU processes

`discretization.discreteCIR(kappa,mu,sigma,Delta,N)` and
`discrete_cir(mean_reversion,long_run_mean,volatility,step,state_count)` select the
same feasible even grid `x_n=d*n`, maximize stationary Gamma coverage within the
sufficient feasibility interval, and match exact conditional mean and variance.
The Feller restriction is `sigma^2 <= 2*kappa*mu`; `N>=3`. An empty feasible set
raises an error suggesting more nodes. Python's optional keyword `method="even"`
selects this same corrected construction. The former `coverage` and exponential
grid interface has been retired; it no longer routes to the incorrect historical
conditional-variance formula.

`discretization.discreteOU(kappa,sigma,Delta,gridOrN)` and
`discrete_ou(mean_reversion,volatility,step,grid_or_count)` discretize the centered
Gaussian OU. Pass `N>=3` for endpoints `+/-sigma/sqrt(2*kappa)*sqrt(N-1)`, or a
strictly increasing fixed grid. Both match exact conditional mean and variance
and reject infeasible grids. The legacy MATLAB `discreteOU` wrapper is in
`Discretize OU/`.

All default process routines, including CIR and OU, use the shared sequential
matcher with moments in conditional-standard-deviation units. Accepted moment
residuals are at most `1e-7` in those units; refinement targets `1e-10`.
Mean and variance feasibility use direct grid bounds, including the minimum
variance from the adjacent bracketing nodes; higher moments use a convex-hull
check. CIR and OU require both moments and raise on failure rather than silently
accepting a lower-order fallback. Explicit 0.1.0 floor options retain the frozen
algorithms and their original tolerances for reproducibility.

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
Optional quadrature methods use the same floor-free stabilization and
sequential matching rules, and also return diagnostics.

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
| `discreteOU` | `discrete_ou` |
| `discreteGaussianMixtureAR` | `discrete_gaussian_mixture_ar` |
| `discreteStochasticVolatilityAR` | `discrete_stochastic_volatility_ar` |
| `momentMatchedDistribution` | `moment_matched_distribution` |
| `maximumEntropyWeights` | `maximum_entropy_weights` |
| `gaussianMixtureQuadrature` | `gaussian_mixture_quadrature` |
| `dataDrivenGaussianQuadrature` | `data_driven_gaussian_quadrature` |
| `cirTransitionDensity` | `cir_transition_density` |
