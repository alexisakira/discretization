# MATLAB optimization results

These measurements compare the same representative workloads before and after
the performance pass. They were collected with MATLAB R2024b after a warm-up
call. The baseline used three repetitions and the optimized run used five;
times are medians and will vary by machine.

| Workload | Before (s) | After (s) | Speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| AR(1), 9 states, 4 moments | 0.48813 | 0.03945 | 12.4x | 91.9% |
| VAR(1), 2D, 81 states | 0.47517 | 0.25676 | 1.85x | 46.0% |
| GMAR(1), 9 states | 0.02204 | 0.00883 | 2.50x | 60.0% |
| GMAR(2), 81 states | 0.55513 | 0.38948 | 1.43x | 29.8% |
| CIR, 9 states | 0.03437 | 0.03292 | 1.04x | 4.2% |
| SV, 45 states | 0.11984 | 0.07532 | 1.59x | 37.2% |

The largest gains came from avoiding optimization problems whose moment targets
are provably infeasible, caching solver configuration, replacing the 2D VAR
rotation optimization with its closed-form solution, and avoiding large
temporary matrices. CIR is already dominated by its small nonlinear solves, so
its measured change is minor.

Output parity is checked against pre-optimization MATLAB fixtures by
`tests/testPerformanceParity.m`.
