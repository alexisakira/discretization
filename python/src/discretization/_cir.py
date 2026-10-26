"""Stable CIR log density, including small-noncentrality and central limits."""

import math

import numpy as np
from scipy.special import gammaln, ive


def cir_log_density(future, current, mean_reversion, long_run_mean, volatility, step):
    c = 2 * mean_reversion / (-math.expm1(-mean_reversion * step) * volatility**2)
    order = 2 * mean_reversion * long_run_mean / volatility**2 - 1
    u = c * current * math.exp(-mean_reversion * step)
    v = c * np.asarray(future, dtype=float)
    result = np.full(v.shape, -np.inf)
    positive = v > 0
    if u == 0:
        result[positive] = (
            math.log(c) - v[positive] + order * np.log(v[positive]) - gammaln(order + 1)
        )
    else:
        argument = 2 * math.sqrt(u) * np.sqrt(v[positive])
        with np.errstate(divide="ignore", invalid="ignore"):
            result[positive] = (
                math.log(c)
                - (math.sqrt(u) - np.sqrt(v[positive])) ** 2
                + order / 2 * (np.log(v[positive]) - math.log(u))
                + np.log(ive(order, argument))
            )
        for index in np.flatnonzero(positive & ~np.isfinite(result)):
            log_product = math.log(u) + math.log(v.flat[index])
            log_term = log_sum = 0.0
            for k in range(1, 100001):
                log_ratio = log_product - math.log(k) - math.log(order + k)
                log_term += log_ratio
                log_sum = max(log_sum, log_term) + math.log1p(
                    math.exp(-abs(log_sum - log_term))
                )
                if log_ratio < 0 and log_term - log_sum < math.log(np.finfo(float).eps):
                    break
            else:
                raise RuntimeError("the CIR log-density series did not converge")
            result.flat[index] = (
                math.log(c)
                - u
                - v.flat[index]
                + order * math.log(v.flat[index])
                - gammaln(order + 1)
                + log_sum
            )
    if order == 0:
        result[v == 0] = math.log(c) - u
    return result
