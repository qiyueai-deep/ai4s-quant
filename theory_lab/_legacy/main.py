"""主线实验：静态精度 + 罕见事件方差 + 平均场条件采样器。"""
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from birthdeath import BirthDeathProcess
from samplers import DiscreteDiffusion
from meanfield import build_dataset, evaluate_conditional, exact_stationary_at
from metrics import (
    tv_distance,
    w1_distance,
    kl_divergence,
    empirical_hist,
    ess_autocorr,
    variance_of_mean_estimate,
)


def static_budget(proc, pi, mcmc_total_budget, sampler, n_seeds=3):
    """在固定总样本预算下，比较 MCMC 与 diffusion 的精度与墙钟。"""
    rows = []
    for n in [5000, 10000, 30000]:
        mcmc_tvs, mcmc_w1s, mcmc_kls, mcmc_wt = [], [], [], []
        diff_tvs, diff_w1s, diff_kls, diff_wt = [], [], [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(s)
            t = time.time()
            samp = proc.mcmc(n_samples=n, rng=rng, burn_in=1000, thin=4)
            mcmc_wt.append(time.time() - t)
            h = empirical_hist(samp, proc.N)
            mcmc_tvs.append(tv_distance(h, pi))
            mcmc_w1s.append(w1_distance(h, pi))
            mcmc_kls.append(kl_divergence(h, pi))

            t = time.time()
            dsamp = sampler.sample(n, rng)
            diff_wt.append(time.time() - t)
            h = empirical_hist(dsamp, proc.N)
            diff_tvs.append(tv_distance(h, pi))
            diff_w1s.append(w1_distance(h, pi))
            diff_kls.append(kl_divergence(h, pi))
        rows.append((
            n,
            np.mean(mcmc_tvs), np.mean(mcmc_w1s), np.mean(mcmc_kls), np.mean(mcmc_wt),
            np.mean(diff_tvs), np.mean(diff_w1s), np.mean(diff_kls), np.mean(diff_wt),
        ))
    return rows


def rare_event_variance(proc, A_mask, n=20000, repeats=20, sampler=None, rng_seed=0):
    """估计 P(X ∈ A) 的方差：MCMC vs diffusion（iid）。"""
    rng = np.random.default_rng(rng_seed)
    # 真实概率
    pi = proc.stationary_exact()
    p_true = float(pi[A_mask].sum())

    mcmc_est, diff_est = [], []
    A_idx = np.where(A_mask)[0]
    for r in range(repeats):
        samp = proc.mcmc(n_samples=n, rng=rng, burn_in=2000, thin=4)
        mcmc_est.append(float(np.mean(np.isin(samp, A_idx))))
    rng = np.random.default_rng(rng_seed + 1)
    for r in range(repeats):
        dsamp = sampler.sample(n, rng)
        diff_est.append(float(np.mean(np.isin(dsamp, A_idx))))
    return p_true, np.var(mcmc_est, ddof=1), np.var(diff_est, ddof=1), mcmc_est, diff_est


def main():
    out_dir = r"D:\混元\2026-09-10-14-27-29\ai4s\results"
    import os
    os.makedirs(out_dir, exist_ok=True)

    # === 静态实验 ===
    proc = BirthDeathProcess(N=64, b0=1.0, b1=0.85, a0=0.6, a1=1.0)
    pi = proc.stationary_exact()
    c_gap = proc.spectral_gap()
    print(f"exact mean = {proc.stationary_mean():.4f}, spectral gap c = {c_gap:.4f}")

    print("\n[1/3] 训练静态扩散采样器 ...")
    rng = np.random.default_rng(0)
    mcmc_train = proc.mcmc(n_samples=30000, rng=rng, burn_in=2000, thin=4)
    diff = DiscreteDiffusion(N=proc.N, hidden=128, T=60)
    t = time.time()
    diff.fit(mcmc_train, steps=4000, lr=1e-3, batch=1024)
    print(f"  训练用时 {time.time()-t:.2f}s")

    print("\n[2/3] 静态精度对照 ...")
    rows = static_budget(proc, pi, 30000, diff, n_seeds=3)
    print(f"  {'N':>6s} | {'MCMC_TV':>9s} {'MCMC_W1':>9s} {'MCMC_KL':>9s} {'MCMC_t':>8s} | {'Diff_TV':>9s} {'Diff_W1':>9s} {'Diff_KL':>9s} {'Diff_t':>8s}")
    for r in rows:
        print(f"  {r[0]:6d} | {r[1]:9.4f} {r[2]:9.4f} {r[3]:9.4f} {r[4]:8.3f} | {r[5]:9.4f} {r[6]:9.4f} {r[7]:9.4f} {r[8]:8.3f}")

    # 保存静态表
    with open(os.path.join(out_dir, "static_budget.tsv"), "w", encoding="utf-8") as f:
        f.write("N\tMCMC_TV\tMCMC_W1\tMCMC_KL\tMCMC_t\tDiff_TV\tDiff_W1\tDiff_KL\tDiff_t\n")
        for r in rows:
            f.write("\t".join(f"{x:.6f}" for x in r) + "\n")

    print("\n[3/3] 罕见事件方差：尾部 i ∈ [40, 64) ...")
    A = np.zeros(proc.N, dtype=bool); A[40:] = True
    p_true, var_m, var_d, mcmc_est, diff_est = rare_event_variance(proc, A, n=20000, repeats=20, sampler=diff, rng_seed=2)
    print(f"  真实 P(i>=40) = {p_true:.5f}")
    print(f"  MCMC  20 次估计方差 = {var_m:.6e}")
    print(f"  Diff  20 次估计方差 = {var_d:.6e}")
    print(f"  方差缩减倍数 = {var_m / max(var_d, 1e-12):.2f}x")

    with open(os.path.join(out_dir, "rare_event.tsv"), "w", encoding="utf-8") as f:
        f.write("p_true\tvar_mcmc\tvar_diff\treduction\n")
        f.write(f"{p_true:.6e}\t{var_m:.6e}\t{var_d:.6e}\t{var_m / max(var_d, 1e-12):.4f}\n")

    # 画直方图对照（精确 vs diffusion vs MCMC）
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    rng = np.random.default_rng(7)
    ds = diff.sample(50000, rng)
    ax[0].bar(np.arange(proc.N) - 0.2, pi, width=0.4, label="exact pi")
    ax[0].bar(np.arange(proc.N) + 0.2, empirical_hist(ds, proc.N), width=0.4, label="diffusion")
    ax[0].set_xlabel("state i"); ax[0].set_ylabel("probability")
    ax[0].set_title("Diffusion samples vs exact stationary"); ax[0].legend()
    ax[0].set_yscale("log")
    ax[1].scatter(mcmc_est, diff_est)
    ax[1].plot([0, p_true * 2.5], [0, p_true * 2.5], "k--", linewidth=0.8)
    ax[1].set_xlabel("MCMC estimate of P(i>=40)")
    ax[1].set_ylabel("Diffusion estimate of P(i>=40)")
    ax[1].set_title(f"Rare-event estimation: variance reduction {var_m / max(var_d, 1e-12):.1f}x")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_distribution_and_rare.png"), dpi=140)
    plt.close(fig)

    # === 平均场条件采样器 ===
    print("\n[4/4] 平均场条件采样器 ...")
    proc_mf = BirthDeathProcess(N=64, b0=1.0, b1=0.85, a0=0.6, a1=1.0, b2=0.3, a2=0.3)
    mu_all = np.linspace(0.0, 8.0, 17)  # 17 个网格点
    # 留出 4 个未见 m 做内插评估
    mu_test = np.array([1.0, 3.0, 5.0, 7.0])
    mu_train = np.array([m for m in mu_all if m not in mu_test])
    rng = np.random.default_rng(3)
    X, C = build_dataset(proc_mf, mu_train, samples_per_mu=4000, rng=rng)
    cond = DiscreteDiffusion(N=proc_mf.N, hidden=128, T=60)
    t = time.time()
    cond.fit(X, conds=C, steps=5000, lr=1e-3, batch=1024)
    print(f"  训练用时 {time.time()-t:.2f}s, 训练集 {len(X)}, 训练点 {len(mu_train)}, 留出点 {len(mu_test)}")
    mf_rows = evaluate_conditional(proc_mf, cond, mu_test, mu_train, n_samples=20000, rng=rng)
    print(f"  {'m_mu':>6s}  {'TV':>8s}  {'W1':>8s}  {'KL':>8s}")
    for r in mf_rows:
        print(f"  {r[0]:6.2f}  {r[1]:8.4f}  {r[2]:8.4f}  {r[3]:8.4f}")

    with open(os.path.join(out_dir, "meanfield_conditional.tsv"), "w", encoding="utf-8") as f:
        f.write("m_mu\tTV\tW1\tKL\n")
        for r in mf_rows:
            f.write(f"{r[0]:.2f}\t{r[1]:.6f}\t{r[2]:.6f}\t{r[3]:.6f}\n")

    # 画条件采样器评估图
    mus = [r[0] for r in mf_rows]
    tvs = [r[1] for r in mf_rows]
    w1s = [r[2] for r in mf_rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(mus, tvs, "o-", label="TV")
    ax[0].plot(mus, w1s, "s-", label="W1")
    ax[0].set_xlabel("mean-field parameter m_mu")
    ax[0].set_ylabel("distance to exact pi(m)")
    ax[0].set_title("Conditional diffusion sampler across m_mu")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    # 精确 pi 在不同 mu 下的均值，用于画整体趋势
    means = [float(np.arange(proc_mf.N) @ exact_stationary_at(proc_mf, m)) for m in mus]
    ax[1].plot(mus, means, "o-")
    ax[1].set_xlabel("m_mu"); ax[1].set_ylabel("E_pi[i]")
    ax[1].set_title("Self-consistent mean as function of m_mu")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_meanfield_conditional.png"), dpi=140)
    plt.close(fig)

    # 谱隙报告
    with open(os.path.join(out_dir, "spectral_gap.tsv"), "w", encoding="utf-8") as f:
        f.write("m_mu\tc_spectral_gap\n")
        for m in mu_test:
            f.write(f"{m:.2f}\t{proc_mf.spectral_gap(m):.6f}\n")

    # 综合结果摘要
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("AI4S 项目 - 分布依赖随机过程生成式采样实验摘要\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"过程参数 (N=64): b0=1.0 b1=0.85 a0=0.6 a1=1.0\n")
        f.write(f"精确均值 = {proc.stationary_mean():.4f}, 谱隙 c = {c_gap:.4f}\n\n")
        f.write("静态精度 (样本量 N): MCMC TV / Diff TV / Diff_t / MCMC_t (秒)\n")
        for r in rows:
            f.write(f"  N={r[0]}: MCMC(TV={r[1]:.4f}, t={r[4]:.3f}s)  Diff(TV={r[5]:.4f}, t={r[8]:.3f}s)\n")
        f.write("\n罕见事件 P(i>=40) 估计 (20 次重复):\n")
        f.write(f"  真实值 = {p_true:.5f}\n")
        f.write(f"  MCMC 估计方差 = {var_m:.4e}\n")
        f.write(f"  Diff 估计方差 = {var_d:.4e}\n")
        f.write(f"  方差缩减倍数 = {var_m / max(var_d, 1e-12):.2f}x\n\n")
        f.write("平均场条件采样器 (b2=a2=0.3):\n")
        f.write("  m_mu    TV      W1      KL\n")
        for r in mf_rows:
            f.write(f"  {r[0]:5.2f}  {r[1]:.4f}  {r[2]:.4f}  {r[3]:.4f}\n")
    print(f"\n[OK] 摘要写入 {summary_path}")


if __name__ == "__main__":
    main()
