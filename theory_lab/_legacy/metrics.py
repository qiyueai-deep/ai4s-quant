"""评估指标：分布距离、有效样本量（ESS）、方差缩减、墙钟。"""
import numpy as np


def tv_distance(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def w1_distance(p, q):
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.abs(cdf_p - cdf_q).sum())


def kl_divergence(p, q, eps=1e-12):
    q = np.clip(q, eps, None)
    return float((p * np.log(np.clip(p, eps, None) / q)).sum())


def empirical_hist(samples, N):
    h = np.bincount(samples, minlength=N).astype(float)
    return h / h.sum()


def ess_autocorr(x):
    """用自相关法估计 ESS（初始正序列法）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    if n < 10 or x.var() == 0:
        return float(n)
    acov = np.correlate(x, x, mode="full")[n - 1:] / (n * x.var())
    rho_sum = 0.0
    for k in range(1, n - 1):
        pair = acov[k] + acov[k + 1] if k + 1 < len(acov) else acov[k]
        if pair > 0:
            rho_sum += pair
        else:
            break
    tau = 1.0 + 2.0 * rho_sum
    return float(n / max(tau, 1.0))


def variance_of_mean_estimate(x, batch=50):
    """Batch means 估计均值的方差（用于方差缩减倍数）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < batch * 4:
        return float(x.var() / max(n, 1))
    nb = n // batch
    means = x[: nb * batch].reshape(nb, batch).mean(axis=1)
    return float(means.var(ddof=1) / nb)
