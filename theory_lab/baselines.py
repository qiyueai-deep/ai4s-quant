"""Reference baselines, including faithful reproductions of the two broken ones.

EnergyMLP is the repaired energy model: log Z must be logsumexp(-E), otherwise
the objective has no finite minimiser. LegacyD3PM reproduces the old sampler
single-step corruption plus uniform-noise re-injection in the reverse chain, so
the regression test can quantify what was wrong.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import make_mlp
from metrics import hist
from process import stationary


class EnergyMLP:
    """p(i) proportional to exp(-E(i)), trained by exact maximum likelihood."""

    def __init__(self, N, hidden=64, seed=0):
        torch.manual_seed(seed)
        self.N = N
        self.emb = nn.Embedding(N, 16)
        self.net = make_mlp(16, hidden, 1, layers=3)

    def energy(self, idx):
        return self.net(self.emb(torch.as_tensor(idx, dtype=torch.long))).squeeze(-1)

    def log_prob(self):
        E = self.energy(torch.arange(self.N))
        return -E - torch.logsumexp(-E, dim=0)

    def fit(self, samples, steps=2000, lr=5e-3, seed=1):
        torch.manual_seed(seed)
        q = torch.as_tensor(hist(samples, self.N), dtype=torch.float32)
        opt = torch.optim.Adam(list(self.emb.parameters()) + list(self.net.parameters()), lr=lr)
        last = float("nan")
        for _ in range(steps):
            logp = self.log_prob()
            loss = -(q * logp).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        return last

    @torch.no_grad()
    def sample(self, n, rng):
        p = self.log_prob().exp().numpy()
        p = np.clip(p, 0.0, None)
        p /= p.sum()
        return rng.choice(self.N, size=n, p=p)


class LegacyD3PM:
    """Old sampler: single-step corruption and noise re-injection at every reverse step."""

    def __init__(self, N, hidden=128, T=60, seed=0):
        torch.manual_seed(seed)
        self.N, self.T = N, T
        self.betas = torch.linspace(1e-4, 0.06, T + 1)
        self.net = make_mlp(N + 2, hidden, N, layers=3)

    def _inp(self, x, t, c):
        xoh = F.one_hot(torch.as_tensor(x, dtype=torch.long), self.N).float()
        return torch.cat([xoh, t[:, None], c[:, None]], dim=1)

    def fit(self, samples, steps=1500, lr=1e-3, batch=1024, seed=1):
        torch.manual_seed(seed)
        x0 = torch.as_tensor(np.asarray(samples), dtype=torch.long)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        for _ in range(steps):
            idx = torch.randint(0, len(x0), (batch,))
            xb = x0[idx]
            t = torch.randint(1, self.T + 1, (batch,))
            beta = self.betas[t]
            corrupt = torch.rand(batch) < beta
            noise = torch.randint(0, self.N, (batch,))
            xt = torch.where(corrupt, noise, xb)
            logits = self.net(self._inp(xt, t.float() / self.T, torch.zeros(batch)))
            loss = F.cross_entropy(logits, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.item())

    @torch.no_grad()
    def sample(self, n, rng):
        x = torch.as_tensor(rng.integers(0, self.N, size=n).astype(np.int64))
        for t in range(self.T, 0, -1):
            logits = self.net(self._inp(x, torch.full((n,), t / self.T), torch.zeros(n)))
            x0_hat = torch.distributions.Categorical(logits=logits).sample()
            beta = float(self.betas[t])
            keep = torch.rand(n) >= beta
            noise = torch.randint(0, self.N, (n,))
            x = torch.where(keep, x0_hat, noise)
        return x.numpy().astype(np.int64)


def hist_iid(samples, N, n, rng):
    return rng.choice(N, size=n, p=hist(samples, N))


def oracle_iid(pi, n, rng):
    return rng.choice(len(pi), size=n, p=pi)


def metropolis_thin(p, m, n, rng, burn_in=2000, thin=4):
    """Old weak baseline: Metropolis with thinning, target pi_m."""
    pi = stationary(p, m)
    x = int(np.argmax(pi))
    out = np.empty(n, dtype=np.int64)
    for t in range(n * thin + burn_in):
        y = x + 1 if rng.random() < 0.5 else x - 1
        y = min(max(y, 0), p.N - 1)
        if rng.random() < min(1.0, pi[y] / pi[x]):
            x = y
        if t >= burn_in and (t - burn_in) % thin == 0:
            out[(t - burn_in) // thin] = x
    return out
