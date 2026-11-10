"""End-to-end experiments. All numbers in README.md come from this file.

    python experiments.py                 # full run
    python experiments.py --seed 7 --scale 0.4   # independent faster replication
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import baselines as B
import coupled as C
import diffusion as D
import inference as I
import process as P
from metrics import bootstrap_ci, ess, hist, kl, nll, tv, w1

OUT = Path(__file__).resolve().parent / "results"


def write_tsv(name, header, rows):
    with (OUT / name).open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(f"{v:.8g}" if isinstance(v, (float, np.floating)) else str(v)
                              for v in r) + "\n")


def write_json(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    with (OUT / name).open("w", encoding="utf-8") as f:
        json.dump(conv(obj), f, ensure_ascii=False, indent=2)


def exp_regression(p, rng, summary):
    """Every fatal defect from the old code, turned into a measurable check."""
    rows = []
    pi5 = P.stationary(p, 5.0)
    Q = P.generator(p, 5.0)
    gap, zero = P.spectral_gap(p, 5.0)
    rows.append(("CTMC balance |Q^T pi|inf", float(np.abs(Q.T @ pi5).max()), "< 1e-12"))
    rows.append(("spectral zero mode |lambda_0|", abs(zero), "< 1e-8"))
    tail, _ = P.truncation_report(p, np.linspace(0, 12, 49))
    rows.append(("truncation tail max pi[N-1]", tail, "< 1e-12"))
    s = P.gillespie_time_uniform(p, 5.0, 50.0 + 30000 * 2.0, rng, dt=2.0, burn_in=50.0)
    e = P.embedded_chain(p, 5.0, 30000, rng)
    tv_ctmc, tv_jump = tv(hist(s, p.N), pi5), tv(hist(e, p.N), pi5)
    rows.append(("TV time-uniform Gillespie", tv_ctmc, "< 0.03"))
    rows.append(("TV legacy embedded jump chain", tv_jump, "> 0.08 (bug)"))
    d1, d2 = P.F_prime_fd(p, 3.0), P.F_prime_poisson(p, 3.0)
    rows.append(("|F'_fd - F'_Poisson|", abs(d1 - d2), "< 1e-6"))
    b = I.derivative_bound(p, 3.0)
    rows.append(("|F'| <= spectral upper bound", abs(b["Fprime_fd"]),
                 f"bound {b['derivative_upper_bound']:.4f}"))

    # EBM normalisation: repaired model integrates to one, old one does not.
    ebm = B.EnergyMLP(p.N, seed=0)
    ebm.fit(s, steps=600, lr=5e-3, seed=1)
    import torch
    new_norm = float(ebm.log_prob().exp().sum().item())
    rows.append(("repaired EBM sum p", new_norm, "= 1"))

    old_norm = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "legacy_samplers", Path(__file__).resolve().parent / "_legacy" / "samplers.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        old = mod.EnergySampler(p.N, hidden=64, seed=0)
        old_norm = float(old.log_prob(torch.arange(p.N)).exp().sum().item())
    except Exception as exc:
        old_norm = f"unavailable: {exc}"
    if isinstance(old_norm, float):
        rows.append(("legacy EBM sum p", old_norm, "far from 1 (bug)"))
    write_tsv("01_regression.tsv", ["check", "value", "criterion"], rows)
    summary["regression"] = dict(
        tv_time_uniform=float(tv_ctmc), tv_embedded_jump=float(tv_jump),
        truncation_tail=float(tail), spectral_gap=float(gap),
        poisson_agreement=float(abs(d1 - d2)),
        derivative_bound=float(b["derivative_upper_bound"]),
        derivative_abs=float(abs(b["Fprime_fd"])),
        ebm_repaired_normalisation=new_norm,
        ebm_legacy_normalisation=old_norm)


def exp_bifurcation(p, summary):
    pbig = p.update(N=128)
    grid = np.arange(8.0, 14.01, 0.25)
    window, counts = P.bifurcation_window(pbig, grid)
    fps = P.fixed_points(p)
    rows = [(float(g), int(c)) for g, c in zip(grid, counts)]
    write_tsv("02_bifurcation.tsv", ["beta", "n_fixed_points"], rows)
    write_tsv("03_fixed_points.tsv", ["m", "Fprime", "gap", "stable"],
              [(r["m"], r["Fprime"], r["gap"], int(r["stable"])) for r in fps])
    # |F'| -> 1 at the edges makes eps/(1-L) blow up: quantitative critical slowing down.
    edge = []
    for beta in list(np.linspace(9.05, 9.35, 7)) + list(np.linspace(10.9, 11.2, 7)):
        q = p.update(beta=float(beta))
        for r in P.fixed_points(q, hi=40):
            L = abs(r["Fprime"])
            edge.append((float(beta), r["m"], L, 1.0 / (1.0 - L) if L < 1 else np.inf))
    write_tsv("04_critical_slowing.tsv", ["beta", "m", "abs_Fprime", "amplification"], edge)

    ms = np.linspace(0, 14, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for beta, style in [(8.0, ":"), (10.0, "-"), (13.0, "--")]:
        q = p.update(beta=beta)
        ax[0].plot(ms, [P.mean(q, m) - m for m in ms], style, label=f"beta={beta}")
    ax[0].axhline(0, color="k", lw=0.7)
    for r in fps:
        ax[0].plot(r["m"], 0, "o" if r["stable"] else "x", ms=7)
    ax[0].set_xlabel("m"); ax[0].set_ylabel("F(m) - m")
    ax[0].set_title("Self-consistency map and fixed points"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(grid, counts, "o-")
    ax[1].set_xlabel("beta"); ax[1].set_ylabel("number of fixed points")
    ax[1].set_title("Bistable window"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_bifurcation.png", dpi=140); plt.close(fig)
    finite_amp = [v[3] for v in edge if np.isfinite(v[3])]
    summary["bifurcation"] = dict(window=window, fixed_points=fps,
                                  max_amplification=float(max(finite_amp)) if finite_amp else None)
    return fps


def exp_kernels(p, summary):
    kp = D.physical_kernel(p, 5.5, T=40)
    ku = D.uniform_kernel(p.N, T=40)
    out = {}
    rows = []
    for k in (kp, ku):
        mx, mn, bd = D.mixing_curve(k)
        out[k.name] = dict(gap=k.gap, t_final=k.t_final, T=k.T,
                           tv_final=float(mx[-1]), bound_final=float(bd[-1]),
                           bound_holds=bool(np.all(mx <= bd + 1e-9)))
        for j in range(0, k.T + 1, 4):
            rows.append((k.name, j, float(k.gap * k.ts[j]), float(mx[j]), float(bd[j])))
    write_tsv("05_kernels.tsv", ["kernel", "step", "mixing_budget_c_times_t", "tv_max", "spectral_bound"], rows)
    weyl = D.weyl_gap_relation(p, 5.5)
    pb = D.physical_gap_bounds(p, 5.5)
    write_tsv("05b_spectral_links.tsv",
              ["gap_R", "gap_Q", "max_diag_delta", "weyl_holds", "min_weight", "gap_lower_bound", "lower_holds"],
              [(weyl["gap_R"], weyl["gap_Q"], weyl["max_diag_delta"],
                int(weyl["weyl_bound_holds"]), pb["min_weight"], pb["lower_bound"],
                int(pb["bound_holds"]))])
    out["weyl"] = weyl
    out["physical_gap_lower_bound"] = pb
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in (kp, ku):
        mx, _, bd = D.mixing_curve(k)
        x = k.gap * k.ts
        ax.semilogy(x, np.maximum(mx, 1e-12), label=f"{k.name} measured")
        ax.semilogy(x, bd, "--", label=f"{k.name} bound")
    ax.set_xlabel("dimensionless mixing budget c * t"); ax.set_ylabel("max TV to uniform")
    ax.set_title("Physical vs uniform corruption: L2 contraction")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_kernels.png", dpi=140); plt.close(fig)
    summary["kernels"] = out


def build_training(p, mu_train, samples_per_mu, rng, dt=2.0):
    X, Mu = [], []
    for m in mu_train:
        T = 50.0 + samples_per_mu * dt
        s = P.gillespie_time_uniform(p, float(m), T, rng, dt=dt, burn_in=50.0)
        X.append(s)
        Mu.append(np.full(len(s), float(m)))
    X = np.concatenate(X)
    Mu = np.concatenate(Mu)
    c_mean, c_std = float(Mu.mean()), float(Mu.std() + 1e-8)
    return X, (Mu - c_mean) / c_std, c_mean, c_std


def make_kernel_of(p, kind, c_mean, c_std, T=40):
    cache = {}
    base_uniform = D.uniform_kernel(p.N, T=T) if kind == "uniform" else None

    def kernel_of(c):
        if kind == "uniform":
            return base_uniform
        key = round(float(c), 8)
        if key not in cache:
            cache[key] = D.physical_kernel(p, float(key) * c_std + c_mean, T=T)
        return cache[key]
    return kernel_of


def exp_generators(p, rng, steps, summary):
    mu_train = np.linspace(0.0, 12.0, 13)
    mu_test = np.array([1.5, 5.5, 9.5])
    X, conds, c_mean, c_std = build_training(p, mu_train, 3000, rng)
    kof_p = make_kernel_of(p, "physical", c_mean, c_std)
    kof_u = make_kernel_of(p, "uniform", c_mean, c_std)
    phys = D.CategoricalDiffusion(p.N, hidden=128, seed=0)
    uni = D.CategoricalDiffusion(p.N, hidden=128, seed=0)
    t0 = time.time()
    lp = phys.fit(kof_p, X, conds, steps=steps, lr=1e-3, batch=1024, seed=1)
    lu = uni.fit(kof_u, X, conds, steps=steps, lr=1e-3, batch=1024, seed=1)
    train_time = time.time() - t0

    rows, detail = [], {}
    n_eval, seeds = 10000, 3
    for m in mu_test:
        c = (float(m) - c_mean) / c_std
        pi = P.stationary(p, float(m))
        for name, model, kof in (("physical", phys, kof_p), ("uniform", uni, kof_u)):
            kern = kof(c)
            tvs, w1s, kls, nlls, times = [], [], [], [], []
            for s in range(seeds):
                r = np.random.default_rng(9000 + s)
                t0 = time.time()
                smp = model.sample(n_eval, kern, c, r)
                times.append(time.time() - t0)
                h = hist(smp, p.N)
                tvs.append(tv(h, pi)); w1s.append(w1(h, pi))
                kls.append(kl(h, pi)); nlls.append(nll(pi, smp))
            lo, hi, _ = bootstrap_ci(np.array(tvs), B=2000, rng=np.random.default_rng(0))
            rows.append((float(m), name, float(np.mean(tvs)), lo, hi,
                         float(np.mean(w1s)), float(np.mean(kls)), float(np.mean(nlls)),
                         float(np.mean(times))))
            x0_m = P.gillespie_time_uniform(p, float(m), 50.0 + 8192 * 2.0,
                                             np.random.default_rng(600 + int(m)),
                                             dt=2.0, burn_in=50.0)
            detail[f"{name}@{m}"] = dict(tv=float(np.mean(tvs)), tv_ci=[lo, hi],
                                         w1=float(np.mean(w1s)), kl=float(np.mean(kls)),
                                         nll=float(np.mean(nlls)),
                                         denoising_ce=float(model.denoising_ce(kern, c, x0_m, rng)),
                                         sample_seconds=float(np.mean(times)))
    write_tsv("06_generators.tsv",
              ["m", "kernel", "TV", "TV_ci_lo", "TV_ci_hi", "W1", "KL", "NLL", "sample_seconds"], rows)

    # Baselines at the held-out m = 5.5, all evaluated identically.
    m0 = 5.5
    pi0 = P.stationary(p, m0)
    train_m0 = P.gillespie_time_uniform(p, m0, 50.0 + 30000 * 2.0, rng, dt=2.0, burn_in=50.0)
    ebm = B.EnergyMLP(p.N, seed=0)
    ebm.fit(train_m0, steps=800, lr=5e-3, seed=1)
    legacy = B.LegacyD3PM(p.N, hidden=128, T=60, seed=0)
    legacy.fit(train_m0, steps=max(400, int(steps * 0.5)), seed=1)
    base_rows = []
    for name, fn in (
        ("oracle iid", lambda r: B.oracle_iid(pi0, n_eval, r)),
        ("histogram iid", lambda r: B.hist_iid(train_m0, p.N, n_eval, r)),
        ("Gillespie CTMC", lambda r: P.gillespie_time_uniform(p, m0, 50 + n_eval * 2.0, r, dt=2.0, burn_in=50.0)),
        ("Metropolis thin=4 (legacy)", lambda r: B.metropolis_thin(p, m0, n_eval, r)),
        ("repaired EBM", lambda r: ebm.sample(n_eval, r)),
        ("legacy D3PM", lambda r: legacy.sample(n_eval, r)),
    ):
        tvs, esss = [], []
        for s in range(seeds):
            r = np.random.default_rng(700 + s)
            smp = fn(r)
            h = hist(smp, p.N)
            tvs.append(tv(h, pi0))
            esss.append(ess(smp.astype(float)))
        base_rows.append((name, float(np.mean(tvs)), float(np.mean(esss))))
    write_tsv("07_baselines.tsv", ["estimator", "TV", "ESS"], base_rows)
    summary["generators"] = dict(train_samples=int(len(X)), steps=steps,
                                 train_seconds=float(train_time),
                                 final_ce_physical=float(lp), final_ce_uniform=float(lu),
                                 detail=detail, baselines=[dict(estimator=a, TV=b, ESS=c)
                                                            for a, b, c in base_rows])
    return phys, kof_p, c_mean, c_std


def exp_self_consistency(p, rng, fps, model, kof, c_mean, c_std, summary):
    stable = [r for r in fps if r["stable"]]
    targets = {round(r["m"], 4): r for r in stable}

    def sampler(n, m_raw, r):
        c = (float(m_raw) - c_mean) / c_std
        return model.sample(n, kof(c), c, r)

    # Empirical 3-sigma bound on the conditional-mean error of the generator.
    n_per_iter = 4000
    eps_list = []
    for m in [1.5, 5.5, 9.5]:
        s = sampler(n_per_iter, m, np.random.default_rng(400 + int(m)))
        eps_list.append(abs(float(s.mean()) - P.mean(p, m)) + 3.0 * s.std() / np.sqrt(len(s)))
    eps_mean = float(max(eps_list))
    rows = []
    cases = []
    for m0, label in [(0.5, "low"), (12.0, "high")]:
        res = I.amortized_fixed_point(sampler, m0, tol=2e-3, max_iter=30,
                                      n_per_iter=n_per_iter,
                                      rng=np.random.default_rng(11 + int(m0)))
        near = min(targets, key=lambda k: abs(k - res["m_hat"]))
        err = abs(res["m_hat"] - near)
        bound = I.propagation_bound(p, near, eps_mean)["bound"]
        rows.append((label, m0, res["m_hat"], near, err, bound, res["iters"], res["wall"]))
        cases.append(dict(label=label, m0=m0, m_hat=res["m_hat"], exact=near,
                          error=err, bound=bound, bound_holds=bool(err <= bound),
                          iters=res["iters"], wall=res["wall"],
                          trajectory=res["trajectory"][:12]))
    m_ref = stable[0]["m"]
    sim = I.simulation_fixed_point(p, m_ref, tol=2e-3, max_iter=30, n_per_iter=n_per_iter,
                                   rng=np.random.default_rng(99))
    write_tsv("08_self_consistency.tsv",
              ["case", "m0", "m_hat", "m_exact", "abs_error", "theory_bound", "iters", "wall_seconds"], rows)
    write_tsv("09_amortization_cost.tsv", ["method", "m_hat", "iters", "wall_seconds"],
              [("amortized", cases[0]["m_hat"], cases[0]["iters"], cases[0]["wall"]),
               ("fresh simulation", sim["m_hat"], sim["iters"], sim["wall"])])
    summary["self_consistency"] = dict(eps_mean_3sigma=eps_mean, cases=cases,
                                       fresh_simulation=dict(m_hat=sim["m_hat"],
                                                             iters=sim["iters"], wall=sim["wall"]),
                                       amortized_wall=float(cases[0]["wall"]),
                                       simulation_wall=float(sim["wall"]),
                                       wall_ratio_amortized_over_simulation=float(
                                           cases[0]["wall"] / max(sim["wall"], 1e-9)))


def exp_rare_event(p, rng, fps, summary):
    m_star = max((r["m"] for r in fps if r["stable"]))
    pi = P.stationary(p, m_star)
    k = 30
    p_true = float(pi[k:].sum())
    train = P.gillespie_time_uniform(p, m_star, 50 + 30000 * 2.0, rng, dt=2.0, burn_in=50.0)
    tail_hits = int((train >= k).sum())
    scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0]
    scan = I.choose_scale(pi, k, scales, pilot_n=4000, rng=np.random.default_rng(5))
    q = I.tilted_proposal(p.N, k, scan["best_scale"])
    n, repeats = 2000, 200
    is_res = I.repeated_is(pi, q, n, repeats, np.random.default_rng(21))

    crude = []
    r2 = np.random.default_rng(22)
    for _ in range(repeats):
        crude.append(float((r2.choice(p.N, size=n, p=pi) >= k).mean()))
    crude = np.asarray(crude)
    thin = []
    r3 = np.random.default_rng(23)
    for _ in range(50):
        s = B.metropolis_thin(p, m_star, n, r3)
        thin.append(float((s >= k).mean()))
    thin = np.asarray(thin)
    hist_p = float((train >= k).mean())

    req_is, _ = I.required_n(pi, q, 0.05)
    req_crude = (1.0 - p_true) / p_true / 0.05 ** 2
    rows = [
        ("plain IS, tilted proposal", is_res["mean"], is_res["bias"], is_res["variance"],
         is_res["mse"], is_res["mean_ess"], is_res["coverage"]),
        ("oracle crude iid", float(crude.mean()), float(crude.mean() - p_true),
         float(crude.var(ddof=1)), float(crude.var(ddof=1) + (crude.mean() - p_true) ** 2),
         float(n), float("nan")),
        ("Metropolis thin=4 (legacy)", float(thin.mean()), float(thin.mean() - p_true),
         float(thin.var(ddof=1)), float(thin.var(ddof=1) + (thin.mean() - p_true) ** 2),
         float("nan"), float("nan")),
        ("histogram iid (train data)", hist_p, -p_true, 0.0, p_true ** 2, float("nan"), float("nan")),
    ]
    write_tsv("10_rare_event.tsv",
              ["estimator", "mean", "bias", "variance", "MSE", "mean_ESS", "CI_coverage"], rows)
    write_tsv("11_tilt_scan.tsv", ["scale", "cv2_weight"], scan["scan"])
    summary["rare_event"] = dict(k=k, p_true=p_true, m_star=float(m_star),
                                 training_tail_hits=tail_hits,
                                 best_tilt_scale=scan["best_scale"], cv2_weight=scan["best_cv2"],
                                 is_mse=is_res["mse"], is_bias=is_res["bias"],
                                 is_coverage=is_res["coverage"], is_ess=is_res["mean_ess"],
                                 crude_mse_theory=float(I.crude_mse(pi, k, n)),
                                 crude_mse_empirical=float(crude.var(ddof=1) + (crude.mean() - p_true) ** 2),
                                 mse_reduction=float((crude.var(ddof=1) + (crude.mean() - p_true) ** 2)
                                                     / is_res["mse"]),
                                 n_for_5pct_is=req_is, n_for_5pct_crude=float(req_crude))


def exp_coupled(rng, steps, summary):
    cp = C.CoupledParams(Nc=24, beta=10.0)
    Q = C.generator_coupled(cp)
    pi, residual = C.stationary_numeric(Q)
    X = C.gillespie_coupled(cp, 50.0 + 40000 * 2.0, rng, dt=2.0, burn_in=50.0)
    cond = np.full(len(X), 0.0, dtype=np.float32)
    gen = C.AutoregressiveSampler(cp, hidden=128, seed=0)
    t0 = time.time()
    gen.fit(X, cond, steps=steps, lr=1e-3, batch=1024, seed=1)
    fit_time = time.time() - t0
    big = gen.sample(200000, 0.0, np.random.default_rng(31))
    h = gen.flat_hist(big)
    tv_gen, w1_gen = tv(h, pi), w1(h, pi)

    xs = np.array([C.decode(cp, s) for s in range(cp.Nc ** cp.K)], dtype=float)
    mm = xs.mean(1)
    thr = next((t for t in range(2, cp.Nc) if pi[mm >= t].sum() <= 5e-3), cp.Nc - 1)
    A = mm >= thr
    p_true = float(pi[A].sum())
    p_hat = float((big.mean(1) >= thr).mean())
    cost = C.cost_scaling([8, 12, 16, 24, 32, 40, 48, 64], cp)
    slope = float(np.polyfit(np.log([r["states"] for r in cost]),
                             np.log([max(r["seconds"], 1e-6) for r in cost]), 1)[0])
    write_tsv("12_coupled.tsv", ["quantity", "value"],
              [("states", cp.Nc ** cp.K), ("numeric_residual", residual),
               ("TV_generator", tv_gen), ("W1_generator", w1_gen),
               ("train_seconds", fit_time), ("threshold", thr),
               ("rare_prob_true", p_true), ("rare_prob_generated", p_hat),
               ("rare_relative_error", abs(p_hat - p_true) / p_true),
               ("cost_exponent_in_states", slope)])
    write_tsv("13_cost_scaling.tsv", ["Nc", "states", "seconds", "residual"],
              [(r["Nc"], r["states"], r["seconds"], r["residual"]) for r in cost])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(np.arange(len(pi)), pi, width=1.0, label="numeric pi")
    ax[0].bar(np.arange(len(h)), h, width=1.0, alpha=.6, label="autoregressive generator")
    ax[0].set_yscale("log"); ax[0].set_xlabel("flat state index")
    ax[0].set_ylabel("probability"); ax[0].set_title(f"TV = {tv_gen:.4f}")
    ax[0].legend(fontsize=8)
    st = np.array([r["states"] for r in cost])
    sec = np.array([r["seconds"] for r in cost])
    ax[1].loglog(st, sec, "o-", label="dense solve")
    ax[1].set_xlabel("number of states Nc^K"); ax[1].set_ylabel("seconds")
    ax[1].set_title(f"Exact inference cost ~ states^{slope:.2f}")
    ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_coupled.png", dpi=140); plt.close(fig)
    summary["coupled"] = dict(states=cp.Nc ** cp.K, residual=float(residual),
                              tv=float(tv_gen), w1=float(w1_gen),
                              fit_seconds=float(fit_time), threshold=int(thr),
                              rare_true=p_true, rare_generated=p_hat,
                              rare_rel_error=float(abs(p_hat - p_true) / p_true),
                              cost_exponent_in_states=slope, cost=cost)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", type=str, default=None,
                    help="optional output directory for an independent replication")
    a = ap.parse_args()
    if a.out:
        cand = Path(a.out)
        OUT = cand if cand.is_absolute() else Path(__file__).resolve().parent / cand
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    p = P.Params()
    steps = max(500, int(2500 * a.scale))
    summary = dict(seed=a.seed, scale=a.scale, N=p.N, params=dict(
        b0=p.b0, b1=p.b1, a0=p.a0, a1=p.a1, beta=p.beta, gamma=p.gamma, kappa=p.kappa))
    t0 = time.time()
    print("[1/7] regression checks ..."); exp_regression(p, rng, summary)
    print("[2/7] bifurcation diagram ..."); fps = exp_bifurcation(p, summary)
    print("[3/7] corruption kernels ..."); exp_kernels(p, summary)
    print("[4/7] conditional generators ...")
    model, kof, c_mean, c_std = exp_generators(p, rng, steps, summary)
    print("[5/7] amortized self-consistency ...")
    exp_self_consistency(p, rng, fps, model, kof, c_mean, c_std, summary)
    print("[6/7] unbiased rare-event estimation ..."); exp_rare_event(p, rng, fps, summary)
    print("[7/7] coupled particle system ..."); exp_coupled(rng, steps, summary)
    summary["total_seconds"] = time.time() - t0
    write_json("summary.json", summary)
    print(f"\nOK  {time.time()-t0:.1f}s  -> {OUT}")
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("regression", "bifurcation", "kernels", "rare_event")},
                     ensure_ascii=False, indent=2, default=float)[:2600])


if __name__ == "__main__":
    main()
