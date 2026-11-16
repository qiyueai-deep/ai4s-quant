"""Offline benchmark: fixed-K sweep, adaptive controllers, correctness.

    python -m specdec.benchmark            # writes artifacts/benchmark.json

All decoding is greedy and deterministic, so the emitted sequences are
reproducible; only wall-clock varies. Wall-clock on a shared CPU drifts
(thermal scaling, background load), so each repeat runs the baseline and
every configuration BACK-TO-BACK, the speedup is computed per repeat and the
median over repeats is reported with its min/max range. A slow machine phase
hits the numerator and denominator of the same repeat together and cancels.
Every speculative run is checked to produce exactly the target's greedy
output.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .core import load_artifacts
from .decode import AdaptiveController, greedy_generate, speculative_generate

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

N_PROMPTS = 24
PROMPT_CHARS = 14
N_NEW = 64
K_GRID = [2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
REPEATS = 5


def build_prompts(tokenizer, n_prompts=N_PROMPTS, prompt_chars=PROMPT_CHARS, seed=0):
    text = (ROOT / "data" / "corpus_val.txt").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if len(ln) >= prompt_chars]
    if not lines:
        raise RuntimeError("no validation lines long enough for a prompt")
    step = max(1, len(lines) // n_prompts)
    return [tokenizer.encode(ln[:prompt_chars], add_bos=True)
            for ln in lines[::step][:n_prompts]]


def _med(xs):
    return float(np.median(xs))


def run_benchmark(target, draft, tokenizer, repeats=REPEATS, verbose=True):
    torch.manual_seed(0)
    prompts = build_prompts(tokenizer)
    reference = [greedy_generate(target, p, N_NEW) for p in prompts]

    def base_once():
        for p in prompts:
            greedy_generate(target, p, N_NEW)

    def config_once(k):
        st = dict(rounds=0, target_forwards=0, draft_forwards=0,
                  accepted=0, proposed=0, k_history=[], match_history=[])
        outs = []
        for p in prompts:
            ids, s = speculative_generate(target, draft, p, N_NEW, k=k)
            outs.append(ids)
            for key in st:
                st[key] += s[key]
        return outs, st

    controllers = {b: AdaptiveController(target, draft, k_grid=range(1, 33),
                                         ewma=0.9, entropy_bucketing=b)
                   for b in (False, True)}

    def adaptive_once(bucketed):
        ctrl = controllers[bucketed]
        for p in prompts:
            speculative_generate(target, draft, p, N_NEW, controller=ctrl)

    # paired repeats: baseline and every configuration in the same sweep
    base_runs = []
    config_runs = {k: [] for k in K_GRID}
    ada_runs = {b: [] for b in (False, True)}
    for _ in range(repeats):
        t0 = time.perf_counter()
        base_once()
        base_runs.append(time.perf_counter() - t0)
        for k in K_GRID:
            t0 = time.perf_counter()
            config_once(k)
            config_runs[k].append(time.perf_counter() - t0)
        for b in (False, True):
            t0 = time.perf_counter()
            adaptive_once(b)
            ada_runs[b].append(time.perf_counter() - t0)

    def snap_of(ctrl):
        s = ctrl.snapshot()
        s.pop("c_block")
        return s

    rows = []
    for k in K_GRID:
        outs, st = config_once(k)
        ratios = sorted(b / c for b, c in zip(base_runs, config_runs[k]))
        match = all(o == r for o, r in zip(outs, reference))
        mh = st["match_history"]
        slots = []
        for j in range(k):
            tested = sum(1 for m in mh if m >= j)
            slots.append(sum(1 for m in mh if m >= j + 1) / max(tested, 1))
        rows.append(dict(draft_k=k, wall_seconds=_med(config_runs[k]),
                         wall_clock_speedup=_med(ratios),
                         speedup_range=[float(ratios[0]), float(ratios[-1])],
                         mean_acceptance_rate=st["accepted"] / max(st["proposed"], 1),
                         mean_emitted_per_round=(st["accepted"] + st["rounds"]) /
                         max(st["rounds"], 1),
                         target_forwards_per_token=st["target_forwards"] /
                         (len(prompts) * N_NEW),
                         slot_acceptance=slots,
                         outputs_match_greedy=bool(match)))
        if verbose:
            print("  K=%2d  speedup %.3f [%.3f, %.3f]  accept %.4f  "
                  "emitted/round %.2f  match %s" % (
                      k, rows[-1]["wall_clock_speedup"],
                      rows[-1]["speedup_range"][0], rows[-1]["speedup_range"][1],
                      rows[-1]["mean_acceptance_rate"],
                      rows[-1]["mean_emitted_per_round"], match), flush=True)

    adaptive_rows = []
    for bucketed in (False, True):
        ctrl = controllers[bucketed]
        # instrumented pass for K/entropy histories (output already verified
        # above by construction, but re-check anyway)
        ent_hist, k_hist = [], []
        orig_choose = ctrl.choose_k

        def choose(h, _orig=orig_choose):
            v = _orig(h)
            k_hist.append(v)
            ent_hist.append(h)
            return v

        ctrl.choose_k = choose
        outs = [speculative_generate(target, draft, p, N_NEW, controller=ctrl)[0]
                for p in prompts]
        ctrl.choose_k = orig_choose
        match = all(o == r for o, r in zip(outs, reference))
        st_tot = dict(rounds=0, target_forwards=0, draft_forwards=0,
                      accepted=0, proposed=0)
        for p in prompts:
            _, s = speculative_generate(target, draft, p, N_NEW, controller=ctrl)
            for kk in st_tot:
                st_tot[kk] += s[kk]
        bucket_of = [ctrl.bucket(h) for h in ent_hist]
        dist = {b: [kv for kv, bo in zip(k_hist, bucket_of) if bo == b]
                for b in ("low", "high")}
        dist = {b: v for b, v in dist.items() if v}
        ratios = sorted(b / c for b, c in zip(base_runs, ada_runs[bucketed]))
        adaptive_rows.append(dict(
            mode=("bucketed" if bucketed else "global"),
            wall_seconds=_med(ada_runs[bucketed]),
            wall_clock_speedup=_med(ratios),
            speedup_range=[float(ratios[0]), float(ratios[-1])],
            outputs_match_greedy=bool(match),
            mean_acceptance_rate=st_tot["accepted"] / max(st_tot["proposed"], 1),
            target_forwards_per_token=st_tot["target_forwards"] /
            (len(prompts) * N_NEW),
            controller=snap_of(ctrl),
            k_mean=float(sum(k_hist) / len(k_hist)),
            k_min=int(min(k_hist)), k_max=int(max(k_hist)),
            k_distribution={b: dict(mean=float(sum(v) / len(v)),
                                    min=int(min(v)), max=int(max(v)),
                                    rounds=len(v))
                            for b, v in dist.items()}))
        if verbose:
            r = adaptive_rows[-1]
            print("  adaptive-%-8s speedup %.3f [%.3f, %.3f]  K mean %.1f "
                  "range [%d, %d]  match %s  acc_low %.3f acc_high %.3f" % (
                      r["mode"], r["wall_clock_speedup"], r["speedup_range"][0],
                      r["speedup_range"][1], r["k_mean"], r["k_min"], r["k_max"],
                      r["outputs_match_greedy"],
                      r["controller"]["acc_low"], r["controller"]["acc_high"]),
                  flush=True)

    return dict(
        environment=dict(torch_version=torch.__version__,
                         threads=torch.get_num_threads(),
                         n_prompts=len(prompts), prompt_chars=PROMPT_CHARS,
                         n_new=N_NEW, repeats=REPEATS,
                         timing="median of paired repeats",
                         target_parameters=target.parameter_count,
                         draft_parameters=draft.parameter_count),
        baseline_seconds=_med(base_runs),
        speculative=rows,
        adaptive=adaptive_rows,
    )


def main():
    torch.set_num_threads(1)
    target, draft, tokenizer = load_artifacts(ART)
    t0 = time.time()
    print("benchmark: %d prompts x %d tokens, K grid %s, %d paired repeats" %
          (N_PROMPTS, N_NEW, K_GRID, REPEATS), flush=True)
    out = run_benchmark(target, draft, tokenizer)
    out["total_seconds"] = time.time() - t0
    (ART / "benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved artifacts/benchmark.json (%.1fs)" % out["total_seconds"])


if __name__ == "__main__":
    main()
