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

`matlab_r2026a_0_2.mat` contains regenerated version 0.2.0 default AR, VAR,
mixture AR, SV, CIR and OU outputs and MATLAB version metadata. Regenerate it
with `addpath('tests'); generateReferenceData`. The original three R2024b files
remain unchanged so the explicit `1e-8` compatibility option is checked against
independent historical data rather than newly generated expected values.

`python_0_1_ar4.npz` records the unmodified Python AR(1) four-moment routine
from commit `2afad78`, evaluated at `(0, .9, .1, 9)` with `moment_count=4`.
It includes source commit and NumPy/SciPy versions. The explicit legacy option
must reproduce this output exactly in that environment. Its pre-existing
maximum difference from the MATLAB R2024b fixture is about `3.3e-6`; the
floor-free defaults use the new common sequential matcher and tighter parity
checks instead.
