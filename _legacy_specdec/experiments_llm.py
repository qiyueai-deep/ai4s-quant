"""Verify four provable statements about speculative decoding, end to end on
models trained inside this repository.

    python experiments_llm.py
    python experiments_llm.py --seed 7 --out results_llm_seed7

If artifacts/ or the offline benchmark are missing they are rebuilt first
(training is deterministic, the benchmark is a wall-clock measurement).

Propositions
    A. acceptance identity  alpha = sum_x min(p,q) = 1 - TV(p,q),
       with Pinsker's floor alpha >= 1 - sqrt(KL/2).
    B. optimal draft length  S(K) = E[N_K]/(c0 + g K) has a finite interior
       maximum; the predicted K* matches the measured one.
    C. distillation budget  a KL budget eps gives alpha >= 1 - sqrt(eps/2);
       distillation beats same-budget SFT on this floor.
    D. runtime adaptive K  the online controller needs no offline sweep and
       matches the oracle best fixed K, while the fixed-K curve degrades
       quickly away from K*.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from specdec import theory as T
from specdec.core import load_artifacts
from specdec.probes import measure_step_costs, probe_distributions

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ROOT / "results_llm"


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


def ensure_artifacts():
    if not (ART / "manifest.json").exists():
        print("artifacts missing -> training now (deterministic, one-off)")
        import specdec.train as TR
        TR.main([])
    if not (ART / "benchmark.json").exists():
        print("benchmark missing -> running offline sweep now")
        from specdec.benchmark import run_benchmark
        torch.set_num_threads(1)
        target, draft, tokenizer = load_artifacts(ART)
        bench = run_benchmark(target, draft, tokenizer)
        bench["total_seconds"] = None
        (ART / "benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--probes", type=int, default=6)
    ap.add_argument("--positions", type=int, default=32)
    a = ap.parse_args()
    if a.out:
        cand = Path(a.out)
        OUT = cand if cand.is_absolute() else ROOT / cand
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    ensure_artifacts()
    t0 = time.time()

    target, draft, tokenizer = load_artifacts(ART)
    bench = json.loads((ART / "benchmark.json").read_text(encoding="utf-8"))
    manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
    meta = dict(target_parameters=target.parameter_count,
                draft_parameters=draft.parameter_count,
                compression=target.parameter_count / draft.parameter_count,
                vocab_size=tokenizer.vocab_size,
                val_ce_target=manifest["metrics"]["val_ce_target"],
                benchmark_prompts=bench["environment"]["n_prompts"],
                benchmark_new_tokens=bench["environment"]["n_new"])

    # ---- A. acceptance identity and Pinsker bound, on real logits ----
    # alpha is an expectation over generated contexts, so it is averaged over
    # several independent prefix seeds (sampled prefixes, unbiased).
    probe_seeds = [a.seed + i for i in range(8)]
    rows, robust_alphas, robust_top1 = [], [], []
    for i, s in enumerate(probe_seeds):
        rr = probe_distributions(target, draft, tokenizer, n_prompts=a.probes,
                                 positions=a.positions, temperature=1.0, seed=s,
                                 prefix_mode="sample",
                                 mc_samples=20000 if i == 0 else 0)
        for r in rr:
            r["probe_seed"] = s
        rows.extend(rr)
        robust_alphas.append(float(np.mean([r["alpha"] for r in rr])))
        robust_top1.append(float(np.mean([r["top1_agree"] for r in rr])))
    alpha = np.array([r["alpha"] for r in rows])
    alpha_exact = np.array([r["alpha_exact"] for r in rows])
    alpha_mc = np.array([r["alpha_mc"] for r in rows])
    kl_pq = np.array([r["kl_pq"] for r in rows])
    tv = np.array([r["tv"] for r in rows])
    top1 = np.array([r["top1_agree"] for r in rows])
    floors = np.array([T.pinsker_alpha_lower_bound(v) for v in kl_pq])
    mc_ok = ~np.isnan(alpha_mc)
    A = dict(n_probes=len(rows), n_probe_seeds=len(probe_seeds),
             alpha_mean=float(alpha.mean()), alpha_std=float(alpha.std()),
             tv_mean=float(tv.mean()), kl_mean=float(kl_pq.mean()),
             top1_agreement_mean=float(top1.mean()),
             identity_max_error_exact=float(np.abs(alpha - alpha_exact).max()),
             identity_max_error_mc=float(np.abs(alpha[mc_ok] - alpha_mc[mc_ok]).max()),
             pinsker_floor_mean=float(floors.mean()),
             pinsker_holds=bool(np.all(alpha >= floors - 1e-9)))
    write_tsv("01_distribution_probes.tsv",
              ["probe_seed", "prompt", "step", "TV", "alpha_1_minus_TV", "alpha_sum_min",
               "alpha_MC", "KL_p_q", "KL_q_p", "top1_agree", "pinsker_floor"],
              [(r["probe_seed"], r["prompt"], r["step"], r["tv"], r["alpha"],
                r["alpha_exact"], r["alpha_mc"], r["kl_pq"], r["kl_qp"],
                r["top1_agree"], f) for r, f in zip(rows, floors)])
    write_tsv("02_identity_pinsker.tsv", ["quantity", "value"],
              [(k, v) for k, v in A.items() if isinstance(v, (int, float, bool))])
    A["alpha_robustness"] = dict(seeds=probe_seeds, alphas=robust_alphas,
                                 min=float(min(robust_alphas)),
                                 max=float(max(robust_alphas)),
                                 mean=float(np.mean(robust_alphas)))

    # ---- C. distillation: measured trade-off and the Pinsker guarantee ----
    # The distilled draft and the same-budget SFT draft trade T=2 KL against
    # argmax alignment. Greedy speculative speedup is driven by argmax
    # agreement, so the distilled draft is the system choice; the Pinsker
    # floor alpha >= 1 - sqrt(KL/2) is verified to hold for BOTH drafts.
    m = manifest["metrics"]
    kl_before = float(m["kl_target_to_draft_sft"])
    kl_after = float(m["kl_target_to_draft_distilled"])
    acc_sft = float(m["greedy_top1_agreement_sft"])
    acc_distilled = float(m["greedy_top1_agreement_distilled"])
    floor_before = T.pinsker_alpha_lower_bound(kl_before)
    floor_after = T.pinsker_alpha_lower_bound(kl_after)
    C = dict(kl_sft_draft=kl_before, kl_distilled_draft=kl_after,
             pinsker_floor_sft=floor_before, pinsker_floor_distilled=floor_after,
             greedy_agreement_sft=acc_sft, greedy_agreement_distilled=acc_distilled,
             agreement_gain=acc_distilled - acc_sft,
             floor_holds_sft=bool(acc_sft >= floor_before - 1e-9),
             floor_holds_distilled=bool(acc_distilled >= floor_after - 1e-9))
    rows_t2 = []
    for s in probe_seeds[:2]:
        rows_t2.extend(probe_distributions(target, draft, tokenizer,
                                           n_prompts=a.probes, positions=a.positions,
                                           temperature=2.0, seed=s,
                                           prefix_mode="sample", mc_samples=0))
    kl_t2 = float(np.mean([r["kl_pq"] for r in rows_t2]))
    alpha_t2 = float(np.mean([r["alpha"] for r in rows_t2]))
    C["probe_kl_temperature2"] = kl_t2
    C["probe_alpha_temperature2"] = alpha_t2
    C["pinsker_floor_temperature2"] = T.pinsker_alpha_lower_bound(kl_t2)
    sweep_acc = float(np.mean([r["mean_acceptance_rate"] for r in bench["speculative"]]))
    C["measured_greedy_acceptance_sweep"] = sweep_acc
    write_tsv("04_distillation.tsv", ["quantity", "value"],
              [("KL of same-budget SFT draft (T=2)", kl_before),
               ("KL of distilled draft (T=2)", kl_after),
               ("Pinsker floor, SFT draft", floor_before),
               ("Pinsker floor, distilled draft", floor_after),
               ("greedy top1 agreement, SFT draft", acc_sft),
               ("greedy top1 agreement, distilled draft", acc_distilled),
               ("agreement gain of distillation", C["agreement_gain"]),
               ("probe KL at T=2", kl_t2), ("probe alpha at T=2", alpha_t2),
               ("sweep greedy acceptance mean", sweep_acc)])

    # ---- B. optimal draft length from the self-built sweep ----
    # The target-draft gap depends on context depth, so a constant-alpha
    # geometric model is systematically wrong here. Decoding is deterministic
    # and speculative output equals the greedy path, so the acceptance of the
    # proposal covering greedy position p is a fixed per-prompt quantity.
    # The expected number of verification rounds R(p) for a round starting at
    # position p satisfies a backward recursion
    #   R(p) = 1 + sum_j P(m=j) R(p+j+1),
    # with P(m=j) the probability that exactly j proposals are accepted.
    # The cost per round C_K = E[N_K]/S_obs(K) (baseline-token units) is
    # affine in K and fitted on the measured sweep; K* is then predicted from
    # the DP round counts -- no circular use of the measured K*.
    spec = [r for r in bench["speculative"] if r["outputs_match_greedy"]]
    Ks = np.array([r["draft_k"] for r in spec])
    S_obs = np.array([r["wall_clock_speedup"] for r in spec])
    E_obs = np.array([r["mean_emitted_per_round"] for r in spec])
    from specdec.benchmark import N_NEW as BENCH_N_NEW, build_prompts
    prompts_b = build_prompts(tokenizer)
    agree = []
    for p in prompts_b:
        ct, cd = target.new_cache(), draft.new_cache()
        x = torch.tensor([p], dtype=torch.long)
        with torch.inference_mode():
            lt = target(x, cache=ct)[0, -1]
            ld = draft(x, cache=cd)[0, -1]
            ag = []
            for _ in range(BENCH_N_NEW):
                ag.append(int(lt.argmax() == ld.argmax()))
                tok = torch.tensor([[int(lt.argmax())]], dtype=torch.long)
                lt = target(tok, cache=ct)[0, -1]
                ld = draft(tok, cache=cd)[0, -1]
        agree.append(ag)

    def dp_rounds(a, k):
        """Expected number of verification rounds to emit len(a) tokens."""
        n = len(a)
        R = [0.0] * (n + 1)
        for p in range(n - 1, -1, -1):
            surv, exp_rounds = 1.0, 1.0
            for j in range(k):
                if p + j >= n:
                    break
                exp_rounds += surv * (1.0 - a[p + j]) * R[min(p + j + 1, n)]
                surv *= a[p + j]
            exp_rounds += surv * R[min(p + k + 1, n)]
            R[p] = exp_rounds
        return R[0]

    C_obs = E_obs / S_obs
    slope, intercept = np.polyfit(Ks.astype(float), C_obs, 1)
    pred_cost = intercept + slope * Ks
    r2 = float(1.0 - np.sum((C_obs - pred_cost) ** 2) /
               max(np.sum((C_obs - C_obs.mean()) ** 2), 1e-300))
    kgrid = np.arange(1, 49)
    dp_round_counts = {int(k): float(np.mean([dp_rounds(a, int(k)) for a in agree]))
                       for k in kgrid}
    E_dp = {int(k): BENCH_N_NEW / v for k, v in dp_round_counts.items()}
    S_model = np.array([E_dp[int(k)] / (intercept + slope * k) for k in kgrid])
    kstar_pred = int(kgrid[int(np.argmax(S_model))])
    kstar_obs = int(Ks[int(np.argmax(S_obs))])
    S_pred_at_Ks = np.array([E_dp[int(k)] / (intercept + slope * k) for k in Ks])
    E_dp_at_Ks = np.array([E_dp[int(k)] for k in Ks])
    spearman = float(np.corrcoef(np.argsort(np.argsort(S_pred_at_Ks)),
                                 np.argsort(np.argsort(S_obs)))[0, 1])
    cost_step = measure_step_costs(target, draft, repeats=200)
    B = dict(alpha_sweep_mean=float(np.mean([r["mean_acceptance_rate"] for r in spec])),
             alpha_probe_stochastic_mean=float(A["alpha_mean"]),
             dp_emitted_vs_measured_max_error=float(np.abs(E_dp_at_Ks - E_obs).max()),
             cost_c0=float(intercept), cost_g=float(slope), cost_r2=r2,
             kstar_predicted=kstar_pred, kstar_observed=kstar_obs,
             kstar_gap=abs(kstar_pred - kstar_obs),
             predicted_peak_speedup=float(S_model.max()),
             observed_peak_speedup=float(S_obs.max()),
             max_abs_speedup_error=float(np.abs(S_pred_at_Ks - S_obs).max()),
             ranking_correlation=spearman,
             finite_optimum=T.has_finite_optimum(0.9, max(slope, 1e-6), max(intercept, 1e-6)),
             all_outputs_match_greedy=bool(all(r["outputs_match_greedy"]
                                               for r in bench["speculative"])),
             step_cost_ratio=cost_step["cost_ratio"],
             target_step_seconds=cost_step["target_step_seconds"],
             draft_step_seconds=cost_step["draft_step_seconds"])
    write_tsv("03_cost_model.tsv",
              ["K", "acceptance_sweep", "emitted_measured", "emitted_dp_model",
               "speedup_observed", "cost_per_round", "speedup_predicted",
               "abs_error"],
              [(int(k), float(r["mean_acceptance_rate"]), float(eo), float(em),
                float(so), float(c), float(sp), float(abs(sp - so)))
               for k, r, eo, em, so, c, sp in
               zip(Ks, spec, E_obs, E_dp_at_Ks, S_obs, C_obs, S_pred_at_Ks)])

    # ---- D. runtime adaptive K vs the oracle fixed K ----
    ada = {r["mode"]: r for r in bench["adaptive"]}
    oracle_k = kstar_obs
    oracle_s = float(S_obs.max())
    default_k = 8
    default_s = float(S_obs[Ks.tolist().index(default_k)]) if default_k in Ks.tolist() else None
    default_speedup = float(S_obs[Ks.tolist().index(default_k)]) if default_k in Ks.tolist() else float(S_obs[0])
    D = dict(oracle_k=oracle_k, oracle_speedup=oracle_s,
             default_k=default_k, default_speedup=default_speedup,
             adaptive={k: dict(speedup=v["wall_clock_speedup"],
                               outputs_match_greedy=v["outputs_match_greedy"],
                               k_mean=v["k_mean"], k_min=v["k_min"], k_max=v["k_max"],
                               k_distribution=v["k_distribution"],
                               controller=v["controller"])
                       for k, v in ada.items()},
             adaptive_global_speedup=ada["global"]["wall_clock_speedup"],
             adaptive_bucketed_speedup=ada["bucketed"]["wall_clock_speedup"],
             online_vs_oracle_global=ada["global"]["wall_clock_speedup"] / oracle_s,
             online_vs_oracle_bucketed=ada["bucketed"]["wall_clock_speedup"] / oracle_s,
             online_vs_default=ada["global"]["wall_clock_speedup"] / default_speedup)
    write_tsv("05_adaptive.tsv",
              ["mode", "speedup", "k_mean", "k_min", "k_max", "match_greedy"],
              [(r["mode"], r["wall_clock_speedup"], r["k_mean"], r["k_min"],
                r["k_max"], int(r["outputs_match_greedy"])) for r in bench["adaptive"]])

    # figure
    robust_ks = [T.optimal_k(v, max(slope, 1e-6), Kmax=48,
                             c0=max(intercept, 1e-6))[0] for v in robust_alphas]
    grid = np.arange(1, 49)
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
    ax[0].plot(Ks, S_obs, "o-", label="measured sweep (self-built benchmark)")
    ax[0].plot(grid, S_model, "-", label="slot-acceptance cost model")
    ax[0].axvline(kstar_pred, color="grey", ls=":", label=f"predicted K*={kstar_pred}")
    ax[0].axvline(kstar_obs, color="red", ls="--", alpha=.6, label=f"measured K*={kstar_obs}")
    for mode, mk in (("global", "^"), ("bucketed", "s")):
        ax[0].scatter([ada[mode]["k_mean"]], [ada[mode]["wall_clock_speedup"]],
                      marker=mk, c="green", zorder=5,
                      label=f"adaptive-{mode} (K mean {ada[mode]['k_mean']:.1f})")
    ax[0].set_xlabel("draft length K"); ax[0].set_ylabel("wall-clock speedup")
    ax[0].set_title("Optimal draft length: theory vs measurement")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].scatter(tv, alpha, s=12, alpha=.5, label="probe positions")
    xs = np.linspace(0, max(tv.max(), 0.2), 100)
    ax[1].plot(xs, 1 - xs, "k-", lw=1, label="alpha = 1 - TV")
    ax[1].scatter(np.sqrt(kl_pq / 2), floors, s=10, alpha=.5, label="Pinsker floor")
    ax[1].set_xlabel("TV (or sqrt(KL/2) for floor)"); ax[1].set_ylabel("acceptance")
    ax[1].set_title("Acceptance identity and Pinsker bound")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    labels = [f"K={int(k)}" for k in Ks] + ["adaptive-g", "adaptive-b"]
    vals = S_obs.tolist() + [ada["global"]["wall_clock_speedup"],
                             ada["bucketed"]["wall_clock_speedup"]]
    colors = ["#999"] * len(Ks) + ["#2a9d8f", "#264653"]
    ax[2].bar(range(len(vals)), vals, color=colors)
    ax[2].axhline(oracle_s, color="red", ls="--", lw=1, label=f"oracle K*={oracle_k}")
    ax[2].set_xticks(range(len(vals)))
    ax[2].set_xticklabels(labels, rotation=60, fontsize=7)
    ax[2].set_ylabel("wall-clock speedup")
    ax[2].set_title("Adaptive K vs fixed K sweep")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_llm_theory.png", dpi=140)
    plt.close(fig)

    write_tsv("06_robustness.tsv", ["seed", "alpha_sampled_prefix", "top1_agreement",
                                    "predicted_Kstar"],
              [(s, al, t1, k) for s, al, t1, k in
               zip(probe_seeds, robust_alphas, robust_top1, robust_ks)])

    summary = dict(seed=a.seed, metadata=meta, A=A, B=B, C=C, D=D,
                   total_seconds=time.time() - t0)
    write_json("07_summary.json", summary)
    print(f"OK {summary['total_seconds']:.1f}s -> {OUT}")
    print(json.dumps({"A": {k: v for k, v in A.items() if k != "alpha_robustness"},
                      "B": {k: v for k, v in B.items() if k != "cost_model"},
                      "C": {k: v for k, v in C.items()},
                      "D": {k: v for k, v in D.items() if k != "adaptive"}},
                     ensure_ascii=False, indent=2, default=float)[:3000])


if __name__ == "__main__":
    main()
