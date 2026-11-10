"""Provable statements about speculative decoding, with no learned quantities.

A. Acceptance identity. Under the rejection rule min(1, p/q) with proposal q
   and target p, the per-position acceptance probability is
       alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q).
   Pinsker's inequality D(p||q) >= 2 TV^2 gives alpha >= 1 - sqrt(D/2).

B. Optimal draft length. With independent per-position acceptance alpha, the
   number of tokens produced by one verification round is
       E[N_K] = sum_{j=0}^{K} alpha^j = (1 - alpha^(K+1)) / (1 - alpha).
   One round costs one target block verification plus K serial draft steps,
   so relative to the one-token autoregressive baseline
       S(K) = E[N_K] / (1 + g K),
   with g = draft_step_cost + target_block_marginal_cost. Maximising S over K
   gives a finite K*, which explains why a very high acceptance rate need not
   give the best wall-clock speedup.

C. Distillation error propagation. A KL budget eps on the draft implies
   alpha >= 1 - sqrt(eps/2) and therefore
   E[N_K] >= (1 - (1 - sqrt(eps/2))^(K+1)) / sqrt(eps/2).
"""
from __future__ import annotations

import numpy as np


def tv_distance(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def kl_divergence(p, q, eps=1e-300):
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


def acceptance_from_tv(p, q):
    """Exact identity alpha = 1 - TV(p, q)."""
    return 1.0 - tv_distance(p, q)


def acceptance_monte_carlo(p, q, rng, n=200000):
    """E_{x ~ q}[min(1, p/q)], an independent check of the identity."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    x = rng.choice(len(q), size=n, p=q / q.sum())
    return float(np.mean(np.minimum(1.0, p[x] / np.maximum(q[x], 1e-300))))


def pinsker_alpha_lower_bound(kl_value):
    """alpha >= 1 - sqrt(KL/2), clipped to the trivial [0, 1] range."""
    return float(np.clip(1.0 - np.sqrt(max(kl_value, 0.0) / 2.0), 0.0, 1.0))


def expected_tokens(alpha, K):
    if alpha >= 1.0 - 1e-12:
        return float(K + 1)
    return float((1.0 - alpha ** (K + 1)) / (1.0 - alpha))


def speedup_model(alpha, K, g, c0=1.0):
    """S(K) = E[N_K] / (c0 + g K). c0 is the fixed per-round overhead.

    With c0 = 1 the model is normalised to one autoregressive target step. In a
    real wall-clock benchmark the fixed term also absorbs prefill, Python
    dispatch and cache management, so it is calibrated from the data.
    """
    return expected_tokens(alpha, K) / (c0 + g * K)


def optimal_k(alpha, g, Kmax=64, c0=1.0):
    ks = np.arange(1, Kmax + 1)
    scores = np.array([speedup_model(alpha, int(k), g, c0) for k in ks])
    return int(ks[int(np.argmax(scores))]), float(scores.max()), scores


def has_finite_optimum(alpha, g, c0=1.0):
    """Provable: for alpha < 1, E[N_K] <= 1/(1-alpha) is bounded while the cost
    grows linearly, so S(K) -> 0 and the maximum is attained at finite K.
    """
    return bool(0.0 <= alpha < 1.0 and g > 0.0 and c0 > 0.0)


def fit_cost_model(alpha, Ks, observed_speedup):
    """Calibrate the affine cost C_K = E[N_K] / S_obs(K) = c0 + g K.

    Returns the fitted coefficients and the R^2 of the cost fit. No parameter is
    fitted to the acceptance rate: alpha comes from the distribution probes.
    """
    Ks = np.asarray(Ks, dtype=float)
    S = np.asarray(observed_speedup, dtype=float)
    E = np.array([expected_tokens(alpha, int(k)) for k in Ks])
    C = E / S
    slope, intercept = np.polyfit(Ks, C, 1)
    pred = intercept + slope * Ks
    r2 = float(1.0 - np.sum((C - pred) ** 2) / max(np.sum((C - C.mean()) ** 2), 1e-300))
    return dict(c0=float(intercept), g=float(slope), r2=r2,
                expected_tokens=E, effective_cost=C)


def distillation_gain(kl_before, kl_after):
    """Change in the Pinsker-guaranteed acceptance floor after distillation."""
    return dict(before=pinsker_alpha_lower_bound(kl_before),
                after=pinsker_alpha_lower_bound(kl_after),
                delta=pinsker_alpha_lower_bound(kl_after) - pinsker_alpha_lower_bound(kl_before))
