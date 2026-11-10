"""Continuous-time distribution-dependent birth-death process (McKean-Vlasov type).

The physical generator Q is treated as known. Its invariant measure pi is the
quantity to be inferred. For fixed mean-field parameter m the chain is linear,
so pi and the L2 spectral gap are available in closed form; that makes every
learned component in this project checkable against an exact answer.

All sampling in this module is continuous-time. The old code sampled the
embedded jump chain (one jump per step), whose stationary law is
nu_i proportional to pi_i * lambda_i instead of pi_i. That bug is reproduced
on purpose in embedded_chain() so the regression test can measure it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Params:
    N: int = 64
    b0: float = 1.0
    b1: float = 0.5
    a0: float = 0.6
    a1: float = 1.5
    beta: float = 10.0
    gamma: float = 0.0
    kappa: float = 5.0
    tail_tol: float = 1e-12

    def update(self, **kw):
        return replace(self, **kw)


def phi(p: Params, m):
    """Saturating Hill coupling phi(m) = m^2 / (kappa^2 + m^2)."""
    m = np.asarray(m, dtype=float)
    return m * m / (p.kappa * p.kappa + m * m)


def dphi(p: Params, m):
    m = np.asarray(m, dtype=float)
    k2 = p.kappa * p.kappa
    return 2.0 * m * k2 / (k2 + m * m) ** 2


def rates(p: Params, m):
    """birth[i]: i -> i+1, death[i]: i -> i-1, length N with reflecting boundaries."""
    i = np.arange(p.N, dtype=float)
    ph = float(phi(p, m))
    birth = p.b0 + p.b1 * i + p.beta * ph
    death = p.a0 + p.a1 * i + p.gamma * ph
    birth[-1] = 0.0
    death[0] = 0.0
    return birth, death


def drates(p: Params, m):
    dp = float(dphi(p, m))
    db = np.full(p.N, p.beta * dp)
    dd = np.full(p.N, p.gamma * dp)
    db[-1] = 0.0
    dd[0] = 0.0
    return db, dd


def generator(p: Params, m):
    birth, death = rates(p, m)
    Q = np.diag(-(birth + death))
    for i in range(p.N - 1):
        Q[i, i + 1] = birth[i]
        Q[i + 1, i] = death[i + 1]
    return Q


def d_generator(p: Params, m):
    """dQ/dm in closed form, used by the Poisson-equation identity."""
    db, dd = drates(p, m)
    Q = np.diag(-(db + dd))
    for i in range(p.N - 1):
        Q[i, i + 1] = db[i]
        Q[i + 1, i] = dd[i + 1]
    return Q


def stationary(p: Params, m):
    """Exact invariant law of the truncated CTMC: pi_n ~ prod_k birth_k/death_(k+1)."""
    birth, death = rates(p, m)
    log_terms = np.log(birth[:-1]) - np.log(death[1:])
    cum = np.concatenate(([0.0], np.cumsum(log_terms)))
    cum -= cum.max()
    pi = np.exp(cum)
    return pi / pi.sum()


def symmetric_generator(p: Params, m):
    """Jacobi matrix S = diag(sqrt(pi)) Q diag(1/sqrt(pi)), symmetric negative semidefinite.

    Off-diagonal entries are sqrt(birth_i * death_(i+1)). This is where the
    physically weighted graph Laplacian used by the diffusion kernel comes from.
    """
    pi = stationary(p, m)
    birth, death = rates(p, m)
    S = np.zeros((p.N, p.N))
    for i in range(p.N - 1):
        w = np.sqrt(birth[i] * death[i + 1])
        S[i, i + 1] = w
        S[i + 1, i] = w
        S[i, i] -= birth[i]
        S[i + 1, i + 1] -= death[i + 1]
    return S, pi


def spectral_gap(p: Params, m):
    """L2 spectral gap of Q and the numerically computed zero eigenvalue."""
    S, _ = symmetric_generator(p, m)
    ev = np.linalg.eigvalsh(S)
    return float(-ev[-2]), float(ev[-1])


def mean(p: Params, m):
    return float(np.arange(p.N) @ stationary(p, m))


def F_prime_fd(p: Params, m, h=1e-5):
    return (mean(p, m + h) - mean(p, m - h)) / (2.0 * h)


def poisson_solve(p: Params, m, rhs):
    """Solve Q g = rhs in the centered space, with pi^T g = 0 as gauge.

    Q has right null vector 1 and left null vector pi, so rhs must satisfy
    pi^T rhs = 0 and the solution is unique only after the gauge is fixed.
    """
    Q = generator(p, m)
    pi = stationary(p, m)
    A = Q.copy()
    A[-1, :] = pi
    b = np.asarray(rhs, dtype=float).copy()
    b[-1] = 0.0
    return np.linalg.solve(A, b)


def F_prime_poisson(p: Params, m):
    """F'(m) = pi (dQ/dm) g, where Q g = -(i - F(m)).

    Derived from pi' Q + pi Q' = 0 and pi' = -pi Q' Q^# (group inverse).
    Used as an independent check of the finite-difference derivative and of the
    bound |F'| <= ||i-F||_{2,pi} ||dQ|| / gap.
    """
    i = np.arange(p.N, dtype=float)
    pi = stationary(p, m)
    g = poisson_solve(p, m, -(i - mean(p, m)))
    return float(pi @ (d_generator(p, m) @ g))


def truncation_report(p: Params, m_grid):
    tails = np.array([float(stationary(p, m)[-1]) for m in m_grid])
    return float(tails.max()), tails


def ensure_truncation(p: Params, m_grid, max_N=4096):
    """Reflecting truncation is only legitimate when the last state is negligible."""
    q = p
    while True:
        tail, _ = truncation_report(q, m_grid)
        if tail < q.tail_tol or q.N >= max_N:
            return q, tail
        q = q.update(N=min(q.N * 2, max_N))


def gillespie_time_uniform(p: Params, m, T, rng, dt=1.0, burn_in=25.0, x0=None):
    """Continuous-time Gillespie, observed on a uniform time grid.

    Time-uniform observation gives samples from pi. Step-uniform observation
    gives samples from the embedded jump chain instead, which is wrong.
    """
    birth, death = rates(p, m)
    total = birth + death
    times = np.arange(burn_in, T, dt)
    if len(times) == 0:
        raise ValueError("T must exceed burn_in")
    out = np.empty(len(times), dtype=np.int64)
    x = int(rng.integers(0, p.N)) if x0 is None else int(x0)
    t = 0.0
    te = t + rng.exponential(1.0 / total[x])
    j = 0
    while j < len(times):
        while te <= times[j]:
            u = rng.random() * total[x]
            x = x + 1 if u < birth[x] else x - 1
            t = te
            te = t + rng.exponential(1.0 / total[x])
        out[j] = x
        j += 1
    return out


def embedded_chain(p: Params, m, n, rng, burn_in=2000):
    """Legacy bug: one forced jump per recorded step, stationary law nu ~ pi*lambda."""
    birth, death = rates(p, m)
    total = birth + death
    x = int(rng.integers(0, p.N))
    out = np.empty(n, dtype=np.int64)
    for t in range(n + burn_in):
        u = rng.random() * total[x]
        x = x + 1 if u < birth[x] else x - 1
        if t >= burn_in:
            out[t - burn_in] = x
    return out


def fixed_points(p: Params, lo=0.0, hi=30.0, n_grid=1501):
    """All roots of F(m) - m, with derivative, spectral gap and stability."""
    grid = np.linspace(lo, hi, n_grid)
    d = np.array([mean(p, m) - m for m in grid])
    roots = []
    for k in range(n_grid - 1):
        if d[k] == 0.0:
            roots.append(grid[k])
        elif d[k] * d[k + 1] < 0.0:
            a, b = grid[k], grid[k + 1]
            fa = mean(p, a) - a
            for _ in range(90):
                c = 0.5 * (a + b)
                fc = mean(p, c) - c
                if fa * fc <= 0.0:
                    b = c
                else:
                    a, fa = c, fc
            roots.append(0.5 * (a + b))
    rows = []
    for r in roots:
        fp = F_prime_fd(p, r)
        gap, _ = spectral_gap(p, r)
        rows.append(dict(m=float(r), Fprime=float(fp), gap=float(gap), stable=bool(abs(fp) < 1.0)))
    return rows


def n_fixed_points(p: Params, **kw):
    return len(fixed_points(p, **kw))


def fixed_point_count(p: Params, beta, m_hi=35.0, n_grid=1201):
    """Number of self-consistent roots at coupling strength beta."""
    return n_fixed_points(p.update(beta=float(beta)), lo=0.0,
                          hi=max(m_hi, 3.0 * beta), n_grid=n_grid)


def bifurcation_window(p: Params, beta_grid, m_hi=35.0, tol=2e-3):
    """Locate the bistable window: two saddle-node bifurcations bound it.

    Multiplicity is not monotone in beta (the high state eventually swallows the
    other two), so a single critical value does not exist. Both edges are
    resolved by bisection and returned.
    """
    grid = np.asarray(beta_grid, dtype=float)
    counts = np.array([fixed_point_count(p, b, m_hi) for b in grid])
    idx = np.where(counts >= 3)[0]
    if len(idx) == 0:
        return None, counts
    i0, i1 = int(idx[0]), int(idx[-1])
    if i0 == 0:
        left = float(grid[0])
    else:
        a, b = float(grid[i0 - 1]), float(grid[i0])
        while b - a > tol:
            c = 0.5 * (a + b)
            if fixed_point_count(p, c, m_hi) >= 3:
                b = c
            else:
                a = c
        left = 0.5 * (a + b)
    if i1 == len(grid) - 1:
        right = float(grid[-1])
    else:
        a, b = float(grid[i1]), float(grid[i1 + 1])
        while b - a > tol:
            c = 0.5 * (a + b)
            if fixed_point_count(p, c, m_hi) >= 3:
                a = c
            else:
                b = c
        right = 0.5 * (a + b)
    return dict(beta_left=left, beta_right=right, width=right - left), counts
