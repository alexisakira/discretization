# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Harden MATLAB Gaussian-mixture AR input validation and tolerate probability
  normalization roundoff.
- Use log-density priors and innovation-scale moment checks on evenly spaced
  grids; expose row-level accuracy and fallback diagnostics as a third output.
- Add tests for rare shocks, high persistence, grid refinement, unit and
  location invariance, Gaussian limits, and invalid inputs. Optional quadrature
  methods and the Python implementation are unchanged numerically.

## 0.1.0 - 2026-08-17

- Add the initial Python implementation of the Farmer-Toda routines.
- Provide Gaussian AR and VAR, Gaussian-mixture AR, stochastic-volatility AR,
  and Cox-Ingersoll-Ross discretization routines.
- Provide maximum-entropy moment matching and reusable quadrature routines.
- Match deterministic MATLAB R2024b reference results.
- Add automated Python and MATLAB tests, documentation, and package metadata.
