"""冒烟：模型 + 训练闭环（MCMC + Diffusion）。"""
import time
import numpy as np
import torch

from birthdeath import BirthDeathProcess
from samplers import DiscreteDiffusion
from metrics import tv_distance, w1_distance, kl_divergence, empirical_hist, ess_autocorr


def main():
    rng = np.random.default_rng(0)
    proc = BirthDeathProcess(N=64, b0=1.0, b1=0.85, a0=0.6, a1=1.0)
    pi = proc.stationary_exact()
    print(f"exact mean = {proc.stationary_mean():.4f}, spectral gap = {proc.spectral_gap():.4f}")
    print(f"pi head = {pi[:8].round(4)}")

    t = time.time()
    mcmc = proc.mcmc(n_samples=30000, rng=rng)
    print(f"MCMC 30k: {time.time()-t:.2f}s, ESS = {ess_autocorr(mcmc):.0f}")
    h = empirical_hist(mcmc, proc.N)
    print(f"  MCMC  vs exact: TV={tv_distance(h, pi):.4f} W1={w1_distance(h, pi):.4f} KL={kl_divergence(h, pi):.4f}")

    t = time.time()
    diff = DiscreteDiffusion(N=proc.N, hidden=128, T=60)
    diff.fit(mcmc, steps=4000, lr=1e-3, batch=1024)
    print(f"Diffusion fit: {time.time()-t:.2f}s")
    s = diff.sample(30000, rng)
    h = empirical_hist(s, proc.N)
    print(f"  Diff  vs exact: TV={tv_distance(h, pi):.4f} W1={w1_distance(h, pi):.4f} KL={kl_divergence(h, pi):.4f}")


if __name__ == "__main__":
    main()
