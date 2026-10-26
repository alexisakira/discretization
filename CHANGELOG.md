# Changelog

All notable changes to this project will be documented in this file.

## 0.2.0 - 2026-10-26

- **Default numerical outputs change.** Remove the absolute `1e-8` reference
  weight floor from Gaussian AR/VAR, Gaussian-mixture AR and stochastic-volatility
  AR in MATLAB and Python. Such a floor can distort rare transitions and accumulate
  material errors in likelihood-based estimation, even when simulated moments look
  accurate. Grid widths and other process defaults are unchanged.
- Construct Gaussian and mixture reference weights in log space and stabilize
  relative to each row's maximum at machine `realmin` (`numpy.finfo(float).tiny`).
  Match moments sequentially, retain the highest successful count, and use a point
  mass nearest the conditional mean when mean matching is infeasible or fails.
  Return row-level moment counts and standardized residuals (per rotated dimension
  for VAR). Diagnostics measure moment accuracy, not transition or likelihood error.
- Recover the 0.1.0 algorithms with MATLAB `'ProbabilityFloor',1e-8` or Python
  `probability_floor=1e-8`. This floors **reference weights**, not final transition
  probabilities. Explicit positive floors select the legacy algorithms and have
  no matched-moment diagnostics. Historical entry points forward these options.
- Align Python even-grid Gaussian-mixture AR with MATLAB, including nonzero
  innovation means, higher-order lag structure, and infeasible-row fallbacks.
- Include the CIR rewrite with an automatically selected strictly feasible even
  grid and corrected exact conditional variance. Python now uses the same default;
  its explicit `coverage` / `method` arguments retain the historical construction.
- Add centered Gaussian OU and OU with geometric aggregate asymmetric-Laplace
  jumps in MATLAB and Python. Both support an automatic or fixed grid and reject
  infeasible two-moment grids. The jump routine is a finite-step splitting model;
  smaller steps approximate continuous jump timing.
- Stabilize the CIR density with scaled Bessel functions and log reference weights;
  include tests, the OU compatibility entry point, regenerated MATLAB R2026a
  default fixtures, and updated guide/API documentation. Preserve the historical
  R2024b fixtures as independent legacy regression checks.

## 0.1.0 - 2026-08-17

- Add the initial Python implementation of the Farmer-Toda routines.
- Provide Gaussian AR and VAR, Gaussian-mixture AR, stochastic-volatility AR,
  and Cox-Ingersoll-Ross discretization routines.
- Provide maximum-entropy moment matching and reusable quadrature routines.
- Match deterministic MATLAB R2024b reference results.
- Add automated Python and MATLAB tests, documentation, and package metadata.
