"""Amortized self-consistency, spectral error bounds and unbiased rare-event IS.

Two theoretical statements are made checkable here.

1. Derivative bound. From pi' Q + pi Q' = 0 and the L2(pi) spectral gap c,
   |F'(m)| <= ||(pi Q')/pi||_{2,pi} * ||i-F||_{2,pi} / c.
   F' is also computed independently by solving the Poisson equation
   Q g = -(i-F), F' = pi (dQ/dm) g.

2. Error propagation. If a learned conditional sampler has conditional-mean
   error at most eps, fixed-point iteration satisfies
   |m_hat - m*| <= eps / (1 - |F'(m*)|).
   Near a saddle-node F' -> 1 and the bound blows up (critical slowing down).
"""
from __future__ import annotations

import time
import numpy as np

from metrics import ci_coverage
from process import (F_prime_fd, F_prime_poisson, d_generator,
                     mean, spectral_gap, stationary)


def derivative_bound(p, m):
    pi = stationary(p, m)
    i = np.arange(p.N, dtype=float)
    F = mean(p, m)
    c, zero = spectral_gap(p, m)
    dQ = d_generator(p, m)
    v = (pi @ dQ) / pi
    norm_v = float(np.sqrt(np.sum(pi * v * v)))
    norm_center = float(np.sqrt(np.sum(pi * (i - F) ** 2)))
    upper = norm_v * norm_center / c
    return dict(m=float(m), Fprime_fd=float(F_prime_fd(p, m)),
                Fprime_poisson=float(F_prime_poisson(p, m)), gap=c,
                zero_mode=zero, norm_v=norm_v, norm_center=norm_center,
                derivative_upper_bound=float(upper))


def propagation_bound(p, m, eps_mean):
    L = abs(F_prime_fd(p, m))
    return dict(m=float(m), L=float(L), eps_mean=float(eps_mean),
                bound=float(eps_mean / (1.0 - L)) if L < 1.0 else float("inf"),
                stable=bool(L < 1.0))


def amortized_fixed_point(sampler, m0, tol=1e-4, max_iter=60, n_per_iter=20000,
                          rng=None, avg_window=5):
    """Stochastic fixed-point iteration using only the learned sampler.

    Polyak averaging of the last avg_window iterates removes the Monte Carlo
    jitter of each individual update; the reported m_hat is the averaged value.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    t0 = time.time()
    m = float(m0)
    traj = [m]
    for _ in range(max_iter):
        s = sampler(n_per_iter, m, rng)
        m_next = float(np.mean(s))
        traj.append(m_next)
        if abs(m_next - m) < tol:
            m = m_next
            break
        m = m_next
    tail = np.asarray(traj[-(avg_window + 1):])
    return dict(m_hat=float(tail.mean()), last_iterate=m,
                trajectory=np.array(traj), iters=len(traj) - 1,
                wall=time.time() - t0)


def simulation_fixed_point(p, m0, tol=1e-4, max_iter=60, n_per_iter=20000,
                           rng=None, avg_window=5):
    """Same iteration, but each F(m) estimate costs a fresh Gillespie simulation."""
    rng = np.random.default_rng(0) if rng is None else rng
    from process import gillespie_time_uniform
    t0 = time.time()
    m = float(m0)
    traj = [m]
    for _ in range(max_iter):
        s = gillespie_time_uniform(p, m, T=n_per_iter * 1.0 + 25.0,
                                   rng=rng, dt=1.0, burn_in=25.0)
        m_next = float(np.mean(s))
        traj.append(m_next)
        if abs(m_next - m) < tol:
            m = m_next
            break
        m = m_next
    tail = np.asarray(traj[-(avg_window + 1):])
    return dict(m_hat=float(tail.mean()), last_iterate=m,
                trajectory=np.array(traj), iters=len(traj) - 1,
                wall=time.time() - t0)


def tilted_proposal(N, k, scale):
    """Full-support proposal on the rare set A = {i >= k}, no training data used."""
    i = np.arange(N)
    q = np.where(i >= k, np.exp(-(np.maximum(i - k, 0)) / scale), 0.0)
    return q / q.sum()


def choose_scale(pi, k, scales, pilot_n, rng):
    """Variance-minimising exponential tilt, chosen on a small pilot."""
    N = len(pi)
    out = []
    for s in scales:
        q = tilted_proposal(N, k, s)
        x = rng.choice(N, size=pilot_n, p=q)
        w = pi[x] / np.maximum(q[x], 1e-300)
        cv2 = float(w.var(ddof=1) / max(w.mean() ** 2, 1e-300))
        out.append((s, cv2))
    best_s, best_cv2 = min(out, key=lambda z: z[1])
    return dict(best_scale=float(best_s), best_cv2=best_cv2, scan=out)


def is_estimate(pi, q, n, rng):
    """Plain importance sampling: unbiased for p_A because q is supported on A."""
    x = rng.choice(len(pi), size=n, p=q)
    w = pi[x] / np.maximum(q[x], 1e-300)
    est = float(w.mean())
    se = float(w.std(ddof=1) / np.sqrt(n))
    ess = float(w.sum() ** 2 / np.sum(w * w))
    return est, se, ess


def repeated_is(pi, q, n, repeats, rng):
    est, se, ess = [], [], []
    for _ in range(repeats):
        e, s, s_eff = is_estimate(pi, q, n, rng)
        est.append(e)
        se.append(s)
        ess.append(s_eff)
    est = np.asarray(est)
    se = np.asarray(se)
    k = int(np.nonzero(q)[0][0])
    truth = float(pi[k:].sum())
    bias = float(est.mean() - truth)
    return dict(truth=truth, mean=float(est.mean()), bias=bias,
                variance=float(est.var(ddof=1)), mse=float(est.var(ddof=1) + bias ** 2),
                mean_se=float(se.mean()), mean_ess=float(np.mean(ess)),
                coverage=ci_coverage(est, se, truth), estimates=est)


def crude_mse(pi, k, n):
    p = float(pi[k:].sum())
    return p * (1.0 - p) / n


def required_n(pi, q, delta):
    """Sample size for relative error delta: n >= (1 + CV_w^2) / delta^2."""
    mask = q > 0
    w = pi[mask] / np.maximum(q[mask], 1e-300)
    p = float(pi[mask].sum())
    cv2 = float(np.sum(q[mask] * (w - p) ** 2) / max(p ** 2, 1e-300))
    return float((1.0 + cv2) / delta ** 2), cv2
