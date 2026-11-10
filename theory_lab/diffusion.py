"""Discrete diffusion with a physically weighted corruption kernel.

Forward corruption is the continuous-time Markov kernel exp(t R), where R is
the symmetric weighted Laplacian of the state path graph. Edge weights
sqrt(b_i a_{i+1}) come from the physical generator Q, not from the invariant
measure pi. Because R is symmetric, the uniform measure is stationary and
exp(t R) drives every initial state to uniform; the L2 contraction rate is the
spectral gap c_R of R, which gives a provable bound on the diffusion horizon.

The reverse step is the exact Bayes posterior

    q(x_{k-1} | x_k, x_0) = Pstep[k][x_{k-1}, x_k] * Pbar[k-1][x_0, x_{k-1}]
                            / Pbar[k][x_0, x_k],

mixed over the network prediction of x_0. No noise is re-injected during the
reverse chain, and the network is trained on x_0 prediction with a proper ELBO
surrogate reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from process import rates, spectral_gap


def expm(A):
    """Scaling-and-squaring Taylor matrix exponential, no SciPy dependency."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    nrm = np.linalg.norm(A, 1)
    s = int(np.ceil(np.log2(max(nrm, 1e-16) / 0.5))) if nrm > 0.5 else 0
    B = A / (2.0 ** s)
    X = np.eye(n) + B
    T = B.copy()
    for k in range(2, 19):
        T = T @ B / k
        X = X + T
    for _ in range(s):
        X = X @ X
    return X


def weighted_laplacian(p, m):
    """Symmetric path-graph Laplacian with physical edge weights sqrt(b_i a_(i+1))."""
    birth, death = rates(p, m)
    w = np.sqrt(birth[:-1] * death[1:])
    R = np.zeros((p.N, p.N))
    for i in range(p.N - 1):
        R[i, i + 1] = w[i]
        R[i + 1, i] = w[i]
        R[i, i] -= w[i]
        R[i + 1, i + 1] -= w[i]
    return R


def graph_gap(R):
    ev = np.linalg.eigvalsh(R)
    return float(-ev[-2]), float(ev[-1])


def weyl_gap_relation(p, m):
    """Link the two spectral gaps used in this project.

    S = diag(sqrt(pi)) Q diag(1/sqrt(pi)) has off-diagonal entries
    w_i = sqrt(b_i a_(i+1)) and diagonal -(b_i+a_i). R has the same
    off-diagonal entries but diagonal -(w_(i-1)+w_i). Hence R - S is diagonal
    and Weyl's inequality gives |c_R - c_Q| <= max_i |(R-S)_ii|.
    """
    R = weighted_laplacian(p, m)
    c_R, _ = graph_gap(R)
    c_Q, _ = spectral_gap(p, m)
    birth, death = rates(p, m)
    w = np.sqrt(birth[:-1] * death[1:])
    wfull = np.zeros(p.N)
    wfull[:-1] = w
    rsum = wfull + np.concatenate(([0.0], wfull[:-1]))
    delta = (birth + death) - rsum
    dmax = float(np.abs(delta).max())
    return dict(gap_R=c_R, gap_Q=c_Q, max_diag_delta=dmax,
                weyl_bound_holds=bool(abs(c_R - c_Q) <= dmax + 1e-9))


def physical_gap_bounds(p, m):
    """c_R >= 2 min_i w_i (1 - cos(pi/N)), from quadratic-form comparison.

    -v^T R v = sum_i w_i (v_i - v_(i+1))^2 >= min_i w_i * v^T L_path v, and
    the second eigenvalue of the unweighted path Laplacian is 2(1-cos(pi/N)).
    The bound ties the physical rates directly to the diffusion horizon.
    """
    R = weighted_laplacian(p, m)
    c_R, _ = graph_gap(R)
    birth, death = rates(p, m)
    w = np.sqrt(birth[:-1] * death[1:])
    lower = 2.0 * float(w.min()) * (1.0 - np.cos(np.pi / p.N))
    return dict(gap_R=c_R, min_weight=float(w.min()), lower_bound=lower,
                bound_holds=bool(lower <= c_R + 1e-12))


def _row_normalise(M):
    M = np.clip(M, 0.0, None)
    return M / np.maximum(M.sum(1, keepdims=True), 1e-300)


@dataclass
class Kernel:
    name: str
    Pbar: np.ndarray        # (T+1, N, N), Pbar[k][x0, xk]
    Pstep: np.ndarray       # (T+1, N, N), Pstep[k][x_(k-1), x_k]
    gap: float              # L2 contraction rate per unit time (or per step)
    t_final: float
    ts: np.ndarray          # cumulative "time" at each k, in gap units

    @property
    def T(self):
        return len(self.Pbar) - 1

    @property
    def N(self):
        return self.Pbar.shape[1]


def physical_kernel(p, m, T=40, eps=1e-3, t_final=None):
    """exp(t R) corruption. Horizon chosen from TV <= 0.5 sqrt(N-1) exp(-c_R t)."""
    R = weighted_laplacian(p, m)
    c, zero = graph_gap(R)
    if t_final is None:
        t_final = float(np.log(np.sqrt(p.N - 1) / (2.0 * eps)) / c)
    dt = t_final / T
    Ps = _row_normalise(expm(dt * R))
    N = p.N
    Pbar = np.empty((T + 1, N, N))
    Pstep = np.empty((T + 1, N, N))
    Pbar[0] = np.eye(N)
    Pstep[0] = np.eye(N)
    for k in range(1, T + 1):
        Pstep[k] = Ps
        Pbar[k] = _row_normalise(Pbar[k - 1] @ Ps)
    return Kernel("physical", Pbar, Pstep, float(c), float(t_final), np.arange(T + 1) * dt)


def uniform_kernel(N, T=40, eps=1e-3):
    """Standard D3PM uniform corruption, calibrated to the same mixing budget.

    Constant beta is chosen so that (1-beta)^T = 2 eps / sqrt(N-1), i.e. the
    same L2 bound eps as physical_kernel with the same number of steps. The
    contraction rate per step is c = -log(1-beta) and ts are step counts, so the
    bound 0.5 sqrt(N-1) exp(-c * ts[k]) is exact for both kernels.
    """
    c = float(np.log(np.sqrt(N - 1) / (2.0 * eps)) / T)
    beta = 1.0 - np.exp(-c)
    Q = (1.0 - beta) * np.eye(N) + beta * np.ones((N, N)) / N
    Pbar = np.empty((T + 1, N, N))
    Pstep = np.empty((T + 1, N, N))
    Pbar[0] = np.eye(N)
    Pstep[0] = np.eye(N)
    for k in range(1, T + 1):
        Pstep[k] = Q
        Pbar[k] = _row_normalise(Pbar[k - 1] @ Q)
    return Kernel("uniform", Pbar, Pstep, c, float(T), np.arange(T + 1, dtype=float))


def mixing_curve(kernel):
    """Measured TV to uniform and the provable L2 bound at each step."""
    N = kernel.N
    u = np.full(N, 1.0 / N)
    tv_max, tv_mean = [], []
    for k in range(kernel.T + 1):
        d = 0.5 * np.abs(kernel.Pbar[k] - u[None, :]).sum(1)
        tv_max.append(float(d.max()))
        tv_mean.append(float(d.mean()))
    bound = 0.5 * np.sqrt(N - 1) * np.exp(-kernel.gap * kernel.ts)
    return np.array(tv_max), np.array(tv_mean), np.array(bound)


def _categorical_rows(probs, rng):
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(len(probs))[:, None]
    return np.asarray((cdf < u).sum(1), dtype=np.int64)


def make_mlp(d_in, hidden, d_out, layers=3):
    mods = [nn.Linear(d_in, hidden), nn.SiLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.SiLU()]
    mods += [nn.Linear(hidden, d_out)]
    return nn.Sequential(*mods)


class CategoricalDiffusion:
    """x_0-prediction network plus an exact Bayes reverse kernel."""

    def __init__(self, N, hidden=128, seed=0):
        torch.manual_seed(seed)
        self.N = N
        self.net = make_mlp(N + 2, hidden, N, layers=3)

    def _features(self, x, t, c):
        x = np.asarray(x, dtype=np.int64)
        xoh = np.eye(self.N, dtype=np.float32)[x]
        return np.concatenate([xoh, np.asarray(t, np.float32)[:, None],
                               np.asarray(c, np.float32)[:, None]], axis=1)

    @torch.no_grad()
    def transition_probs(self, xk, k, kernel, cond):
        """p_theta(x_(k-1) | x_k) after mixing the exact posterior over x_0."""
        xk = np.asarray(xk, dtype=np.int64)
        n = len(xk)
        t = np.full(n, k / kernel.T, dtype=np.float32)
        c = np.full(n, float(cond), dtype=np.float32)
        logits = self.net(torch.from_numpy(self._features(xk, t, c))).numpy()
        logits -= logits.max(1, keepdims=True)
        q0 = np.exp(logits)
        q0 /= q0.sum(1, keepdims=True)
        A = kernel.Pstep[k]                 # A[r, s]
        B = kernel.Pbar[k - 1]              # B[j, r]
        D = kernel.Pbar[k]                  # D[j, s]
        denom = np.clip(D[:, xk].T, 1e-12, None)
        V = (q0 / denom) @ B
        probs = np.clip(V * A[:, xk].T, 0.0, None)
        return _row_normalise(probs)

    def fit(self, kernel_of, x0, conds, steps=3000, lr=1e-3, batch=1024, seed=1):
        """kernel_of(cond) returns a Kernel; conds are the normalized conditions."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        x0 = np.asarray(x0, dtype=np.int64)
        conds = np.asarray(conds, dtype=np.float32)
        T = kernel_of(conds[0]).T
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        last = float("nan")
        for _ in range(steps):
            idx = rng.integers(0, len(x0), batch)
            xx, cc = x0[idx], conds[idx]
            k = rng.integers(1, T + 1, batch)
            xt = np.empty(batch, dtype=np.int64)
            for cv in np.unique(cc):
                mask = cc == cv
                ker = kernel_of(float(cv))
                rows = ker.Pbar[k[mask], xx[mask], :]
                xt[mask] = _categorical_rows(rows, rng)
            feat = self._features(xt, k / T, cc)
            logits = self.net(torch.from_numpy(feat))
            loss = F.cross_entropy(logits, torch.from_numpy(xx.astype(np.int64)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        return last

    @torch.no_grad()
    def sample(self, n, kernel, cond, rng):
        x = rng.integers(0, self.N, size=n).astype(np.int64)
        for k in range(kernel.T, 0, -1):
            probs = self.transition_probs(x, k, kernel, cond)
            x = _categorical_rows(probs, rng)
        return x

    @torch.no_grad()
    def denoising_ce(self, kernel, cond, x0, rng, n=8192, seed=0):
        """Monte Carlo estimate of E_q[-log p_theta(x_0 | x_k)], the VB surrogate."""
        r = np.random.default_rng(seed)
        idx = r.integers(0, len(x0), n)
        xx = np.asarray(x0, dtype=np.int64)[idx]
        k = r.integers(1, kernel.T + 1, n)
        rows = kernel.Pbar[k, xx, :]
        xt = _categorical_rows(rows, r)
        feat = self._features(xt, k / kernel.T, np.full(n, float(cond), np.float32))
        logits = self.net(torch.from_numpy(feat)).numpy()
        logits -= logits.max(1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        return float(-np.mean(logp[np.arange(n), xx]))
