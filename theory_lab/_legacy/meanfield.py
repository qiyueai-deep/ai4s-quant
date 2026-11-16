"""平均场条件采样器：扫 m_mu 网格，训练条件扩散，验证对未见 m_mu 的泛化。"""
import time
import numpy as np
import torch

from birthdeath import BirthDeathProcess
from samplers import DiscreteDiffusion
from metrics import tv_distance, w1_distance, kl_divergence, empirical_hist


def build_dataset(proc, mu_grid, samples_per_mu, rng):
    """对每个 mu 生成 MCMC/Gillespie 训练样本。"""
    all_x, all_c = [], []
    mu_norm = (mu_grid - mu_grid.mean()) / (mu_grid.std() + 1e-8)
    for m, cn in zip(mu_grid, mu_norm):
        b, a = proc.rates(m)
        x = np.empty(samples_per_mu, dtype=np.int64)
        state = int(rng.integers(0, proc.N))
        for t in range(samples_per_mu):
            x[t] = state
            lam = b[state] if state < proc.N - 1 else 0.0
            mu = a[state - 1] if state > 0 else 0.0
            tot = lam + mu
            if tot <= 0:
                continue
            r = rng.random() * tot
            state = min(state + 1, proc.N - 1) if r < lam else max(state - 1, 0)
        # burn-in 一半
        keep = x[len(x) // 2 :]
        all_x.append(keep)
        all_c.append(np.full(len(keep), cn, dtype=np.float32))
    return np.concatenate(all_x), np.concatenate(all_c)


def exact_stationary_at(proc, m):
    """固定 m 时退化为线性生灭过程，π_n = π_0 Π b_k / a_{k+1}。"""
    b, a = proc.rates(m)
    lt = np.log(np.maximum(b, 1e-12)) - np.log(np.maximum(a, 1e-12))
    cum = np.concatenate([[0.0], np.cumsum(lt)])
    cum -= cum.max()
    pi = np.exp(cum)
    return pi / pi.sum()


def evaluate_conditional(proc, cond_sampler, mu_grid_test, mu_train, n_samples, rng):
    """在未见 mu 处评估条件生成器 vs 该 mu 处的精确 π。"""
    mu_mean = mu_train.mean()
    mu_std = mu_train.std() + 1e-8
    rows = []
    for m in mu_grid_test:
        cn = (m - mu_mean) / mu_std
        pi = exact_stationary_at(proc, m)
        s = cond_sampler.sample(n_samples, rng, cond=cn)
        h = empirical_hist(s, proc.N)
        rows.append((m, tv_distance(h, pi), w1_distance(h, pi), kl_divergence(h, pi)))
    return rows


def main():
    rng = np.random.default_rng(1)
    # 启用分布依赖项，b2 = a2 = 0.3
    proc = BirthDeathProcess(N=64, b0=1.0, b1=0.85, a0=0.6, a1=1.0, b2=0.3, a2=0.3)
    mu_train = np.linspace(0.0, 8.0, 12)
    mu_test = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])

    print("构建训练集 ...")
    t0 = time.time()
    X, C = build_dataset(proc, mu_train, samples_per_mu=4000, rng=rng)
    print(f"  样本 {len(X)} 用时 {time.time()-t0:.2f}s, 条件范围 {C.min():.2f}..{C.max():.2f}")

    print("训练条件扩散 ...")
    t0 = time.time()
    diff = DiscreteDiffusion(N=proc.N, hidden=128, T=60)
    diff.fit(X, conds=C, steps=4000, lr=1e-3, batch=1024)
    print(f"  训练用时 {time.time()-t0:.2f}s")

    print("在未见 m_mu 处评估条件生成器 vs 精确 π(m) ...")
    rows = evaluate_conditional(proc, diff, mu_test, mu_train, n_samples=10000, rng=rng)
    print(f"  {'m_mu':>6s}  {'TV':>8s}  {'W1':>8s}  {'KL':>8s}")
    for r in rows:
        print(f"  {r[0]:6.2f}  {r[1]:8.4f}  {r[2]:8.4f}  {r[3]:8.4f}")
    return rows


if __name__ == "__main__":
    main()
