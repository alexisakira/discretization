# MATLAB reference data

`matlab_r2024b_baseline.mat` records deterministic outputs from the original
maximum-entropy implementation on MATLAB R2024b. The automated tests use a
numerical tolerance rather than requiring bit-for-bit agreement, because solver
details can vary across MATLAB releases and between MATLAB and Python.

The fixture currently covers the maximum-entropy core and the Gaussian AR(1)
process. It is intended to anchor the first Python port.

`matlab_r2024b_milestone2.mat` covers reusable quadrature rules,
nonparametric discretization, and both CIR grid methods. It anchors the second
Python milestone.

`matlab_r2024b_performance.mat` freezes representative AR, VAR, Gaussian-mixture
AR, and stochastic-volatility outputs before the MATLAB performance pass. It
also anchors the corresponding Python model ports.
