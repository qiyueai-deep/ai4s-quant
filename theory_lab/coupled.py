"""Interacting-particle mean-field system with no closed-form invariant law.

With K particles on {0..Nc-1} the state space has Nc^K states. The generator
is still known (physics), but its invariant measure is not available by the
one-dimensional birth-death recursion. It can only be obtained by solving a
Nc^K x Nc^K linear system, whose cost grows as Nc^(3K), or by simulation.
This is where amortized generation stops being a toy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from diffusion import make_mlp
from process import phi


@dataclass(frozen=True)
class CoupledParams:
    Nc: int = 24
    K: int = 2
    b0: float = 1.0
    b1: float = 0.5
    a0: float = 0.6
    a1: float = 1.5
    beta: float = 10.0
    gamma: float = 0.0
    kappa: float = 5.0

    def update(self, **kw):
        from dataclasses import replace
        return replace(self, **kw)


def decode(cp, s):
    x = np.empty(cp.K, dtype=np.int64)
    for j in range(cp.K):
        x[j] = s % cp.Nc
        s //= cp.Nc
    return x


def encode(cp, x):
    s = 0
    for j in range(cp.K):
        s += int(x[j]) * cp.Nc ** j
    return s


def _rates(cp, x):
    m = float(x.mean())
    ph = float(phi(cp, m))
    i = np.arange(cp.Nc, dtype=float)
    birth = cp.b0 + cp.b1 * i + cp.beta * ph
    death = cp.a0 + cp.a1 * i + cp.gamma * ph
    birth[-1] = 0.0
    death[0] = 0.0
    return birth, death


def generator_coupled(cp):
    M = cp.Nc ** cp.K
    Q = np.zeros((M, M))
    for s in range(M):
        x = decode(cp, s)
        birth, death = _rates(cp, x)
        for j in range(cp.K):
            if birth[x[j]] > 0:
                Q[s, s + cp.Nc ** j] += birth[x[j]]
                Q[s, s] -= birth[x[j]]
            if death[x[j]] > 0:
                Q[s, s - cp.Nc ** j] += death[x[j]]
                Q[s, s] -= death[x[j]]
    return Q


def stationary_numeric(Q):
    """Exact numerical invariant law; residual is reported, not assumed."""
    M = Q.shape[0]
    A = Q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(M)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0.0, None)
    pi /= pi.sum()
    residual = float(np.abs(Q.T @ pi).max())
    return pi, residual


def gillespie_coupled(cp, T, rng, dt=1.0, burn_in=25.0):
    """Continuous-time simulation observed on a uniform time grid."""
    times = np.arange(burn_in, T, dt)
    out = np.empty((len(times), cp.K), dtype=np.int64)
    x = rng.integers(0, cp.Nc, size=cp.K).astype(np.int64)
    t = 0.0

    def total():
        birth, death = _rates(cp, x)
        return float((birth[x] + death[x]).sum())

    te = t + rng.exponential(1.0 / total())
    j = 0
    while j < len(times):
        while te <= times[j]:
            birth, death = _rates(cp, x)
            r = np.concatenate((birth[x], death[x]))
            cum = np.cumsum(r)
            u = rng.random() * cum[-1]
            a = int(np.searchsorted(cum, u))
            if a < cp.K:
                x[a] += 1
            else:
                x[a - cp.K] -= 1
            t = te
            te = t + rng.exponential(1.0 / total())
        out[j] = x
        j += 1
    return out


class AutoregressiveSampler:
    """p_theta(x_1..x_K | beta) = prod_j p_theta(x_j | x_<j, beta), iid sampling."""

    def __init__(self, cp, hidden=128, seed=0):
        torch.manual_seed(seed)
        self.cp = cp
        self.heads = [make_mlp(cp.K + 1, hidden, cp.Nc, layers=2) for _ in range(cp.K)]

    def _features(self, x, cond, j, n):
        feat = np.zeros((n, self.cp.K + 1), dtype=np.float32)
        if j > 0:
            feat[:, :j] = np.asarray(x[:, :j], np.float32) / (self.cp.Nc - 1)
        feat[:, -1] = cond
        return feat

    def fit(self, X, cond, steps=2500, lr=1e-3, batch=1024, seed=1):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.int64)
        cond = np.asarray(cond, dtype=np.float32)
        params = [q for h in self.heads for q in h.parameters()]
        opt = torch.optim.Adam(params, lr=lr)
        last = float("nan")
        for _ in range(steps):
            idx = rng.integers(0, len(X), batch)
            xb, cb = X[idx], cond[idx]
            loss = 0.0
            for j, head in enumerate(self.heads):
                logits = head(torch.from_numpy(self._features(xb, cb, j, batch)))
                loss = loss + F.cross_entropy(logits, torch.from_numpy(xb[:, j].astype(np.int64)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        return last

    @torch.no_grad()
    def sample(self, n, cond, rng):
        x = np.zeros((n, self.cp.K), dtype=np.int64)
        cb = np.full(n, float(cond), dtype=np.float32)
        for j, head in enumerate(self.heads):
            logits = head(torch.from_numpy(self._features(x, cb, j, n))).numpy()
            logits -= logits.max(1, keepdims=True)
            pr = np.exp(logits)
            pr /= pr.sum(1, keepdims=True)
            cdf = np.cumsum(pr, axis=1)
            cdf[:, -1] = 1.0
            x[:, j] = (cdf < rng.random(n)[:, None]).sum(1)
        return x

    def flat_hist(self, X):
        powers = self.cp.Nc ** np.arange(self.cp.K, dtype=np.int64)
        s = np.asarray(X, dtype=np.int64) @ powers
        h = np.bincount(s, minlength=self.cp.Nc ** self.cp.K).astype(float)
        return h / h.sum()


def cost_scaling(Nc_list, cp, repeats=1):
    rows = []
    for Nc in Nc_list:
        c = cp.update(Nc=Nc)
        t0 = time.time()
        Q = generator_coupled(c)
        pi, res = stationary_numeric(Q)
        rows.append(dict(Nc=Nc, states=c.Nc ** c.K, seconds=time.time() - t0,
                         residual=res))
    return rows
