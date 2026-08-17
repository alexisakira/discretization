# MATLAB performance benchmarks

Run the benchmark suite from the repository root:

```matlab
addpath(genpath(pwd))
benchmarkMatlab(3)
```

The suite warms up each case before recording repeated measurements. Runtime
results are machine-specific; numerical checksums help detect unintended output
changes between benchmark runs.

See [RESULTS.md](RESULTS.md) for the MATLAB R2024b measurements from the
optimization pass. The accompanying `matlab_r2024b_before.mat` and
`matlab_r2024b_after.mat` files contain the raw samples and metadata.
