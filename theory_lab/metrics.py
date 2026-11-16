"""Distribution distances, likelihoods, uncertainty and MCMC efficiency.

Every headline number in this project is reported with a confidence interval
and with the effective sample size behind it. A variance reduction without a
bias / MSE / ESS statement is not a result.
"""
from __future__ import annotations

import numpy as np


def hist(samples, N):
    h = np.bincount(np.asarray(samples, dtype=np.int64), minlength=N).astype(float)
    return h / h.sum()


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def w1(p, q):
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


def kl(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    return float((p * np.log(np.clip(p, eps, None) / q)).sum())


def js(p, q):
    m = 0.5 * (np.asarray(p) + np.asarray(q))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def nll(p, samples, eps=1e-300):
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    return float(-np.mean(np.log(p[np.asarray(samples, dtype=np.int64)])))


def bootstrap_ci(values, B=4000, alpha=0.05, rng=None, stat=np.mean):
    """Percentile bootstrap for a 1-D sample. stat must accept an axis keyword."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0) if rng is None else rng
    idx = rng.integers(0, len(values), size=(B, len(values)))
    boot = stat(values[idx], axis=1)
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi), float(stat(values))


def _autocovariance(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    if acf[0] <= 0:
        return np.zeros(n)
    return acf / acf[0]


def integrated_time(x):
    """Initial positive sequence estimate of tau_int = 1 + 2 sum_k rho_k."""
    rho = _autocovariance(np.asarray(x, dtype=float))
    tau = 1.0
    for k in range(1, len(rho) - 1):
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        tau += pair
    return float(max(tau, 1.0))


def ess(x):
    return float(len(x) / integrated_time(x))


def asymptotic_variance(x):
    """sigma^2_asym of the sample mean times n, i.e. var(x) * tau_int."""
    x = np.asarray(x, dtype=float)
    return float(x.var(ddof=1) * integrated_time(x))


def ci_coverage(estimates, standard_errors, truth, z=1.96):
    estimates = np.asarray(estimates, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    return float(np.mean(np.abs(estimates - truth) <= z * se))


def relative_error(estimates, truth):
    estimates = np.asarray(estimates, dtype=float)
    return float(np.sqrt(np.mean((estimates - truth) ** 2)) / abs(truth)) if truth != 0 else float("inf")
