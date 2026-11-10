"""Quantization experiments on the self-trained testbed model, end to end.

    python experiments_quant.py
    python experiments_quant.py --seed 7 --out results_quant_seed7

E1  frontier: RTN bits x granularity (+6/8 bits for the flat region)
E2  method comparison at 2/3/4 bits: RTN, RTN+clip, GPTQ, GPTQ+clip
E3  theory parity: closed-form predicted reconstruction MSE vs measured for
    RTN and RTN+clip on every layer; GPTQ surrogate identity; measured dCE
    vs predicted total MSE across the grid
E4  water-filling mixed precision vs uniform bits at equal memory, with the
    KKT marginal ratio
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

from lmlab.model import load_checkpoint, CharTokenizer
from quant.calibration import collect_calibration, quantizable_linears
from quant.gptq import gptq_quantize
from quant.pipeline import eval_ce, quantize_model
from quant import theory as T
from quant.rtn import quantize_rows, quantize_rows_clip

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ROOT / "results_quant"


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


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.out:
        cand = Path(a.out)
        OUT = cand if cand.is_absolute() else ROOT / cand
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    t0 = time.time()

    model = load_checkpoint(ART / "target_q.pt")
    tokenizer = CharTokenizer.load(ART / "tokenizer_q.json")
    base = eval_ce(model, tokenizer, seed=a.seed)
    print(f"base CE {base['ce']:.4f}  ppl {base['perplexity']:.2f}", flush=True)

    calib = collect_calibration(model, tokenizer, seed=a.seed)
    layer_names = [name for name, _ in quantizable_linears(model)]
    print(f"calibration done: {len(layer_names)} layers", flush=True)

    # ---- E1: RTN frontier ----
    E1 = []
    for bits in (2, 3, 4, 6, 8):
        for gran in ("per_tensor", "per_channel"):
            qm, _ = quantize_model(model, bits, "rtn", gran)
            ev = eval_ce(qm, tokenizer, seed=a.seed)
            E1.append(dict(bits=bits, granularity=gran, ce=ev["ce"],
                           ppl=ev["perplexity"], dce=ev["ce"] - base["ce"]))
            print("  RTN b=%d %-11s dCE %+.4f ppl %.2f" % (
                bits, gran, E1[-1]["dce"], E1[-1]["ppl"]), flush=True)
    write_tsv("01_rtn_frontier.tsv",
              ["bits", "granularity", "ce", "ppl", "dce"],
              [(r["bits"], r["granularity"], r["ce"], r["ppl"], r["dce"])
               for r in E1])

    # ---- E2: method comparison ----
    E2 = []
    METHODS = (("rtn", False), ("rtn", True), ("gptq", False), ("gptq", True))
    for bits in (2, 3, 4):
        for method, clip in METHODS:
            qm, _ = quantize_model(model, bits, method, "per_channel",
                                   calibration=calib, clip=clip)
            ev = eval_ce(qm, tokenizer, seed=a.seed)
            E2.append(dict(bits=bits, method=method + ("+clip" if clip else ""),
                           ce=ev["ce"], ppl=ev["perplexity"],
                           dce=ev["ce"] - base["ce"]))
            print("  b=%d %-10s dCE %+.4f ppl %.2f" % (
                bits, E2[-1]["method"], E2[-1]["dce"], E2[-1]["ppl"]), flush=True)
    write_tsv("02_methods.tsv",
              ["bits", "method", "ce", "ppl", "dce"],
              [(r["bits"], r["method"], r["ce"], r["ppl"], r["dce"])
               for r in E2])

    # ---- E3: theory parity ----
    parity = []
    for bits in (2, 3, 4):
        for name, module in quantizable_linears(model):
            W = module.weight.data
            st = calib[name]
            # plain RTN prediction and measurement
            pred0 = T.predicted_recon_mse(W, bits, st.H, "per_channel")
            Wq0, _ = quantize_rows(W, bits)
            meas0 = T.measured_recon_mse(W, Wq0, st.H)
            parity.append(dict(layer=name, bits=bits, method="rtn",
                               predicted_mse=pred0["predicted_mse"],
                               measured_mse=meas0,
                               clip_fraction=pred0["clip_fraction"]))
            # clip-search prediction and measurement
            Wq1, scales1, pct1 = quantize_rows_clip(W, bits, st.H)
            pred1 = T.predicted_recon_mse(W, bits, st.H, scales=scales1)
            meas1 = T.measured_recon_mse(W, Wq1, st.H)
            parity.append(dict(layer=name, bits=bits, method="rtn+clip",
                               predicted_mse=pred1["predicted_mse"],
                               measured_mse=meas1,
                               clip_fraction=pred1["clip_fraction"]))
    logratios = [np.log(max(p["measured_mse"], 1e-300) /
                        max(p["predicted_mse"], 1e-300)) for p in parity]
    worst_logratio = float(max(abs(l) for l in logratios))
    print("  parity: worst |log(measured/predicted)| = %.3f over %d layer-bits"
          % (worst_logratio, len(parity)), flush=True)

    # GPTQ surrogate identity: sum (w-q)^2 / U[i,i]^2 == trace(D^T D Hd)
    surr = []
    for bits in (2, 3, 4):
        for name, module in quantizable_linears(model):
            W = module.weight.data
            st = calib[name]
            Hd = st.H.double() + 0.01 * torch.diag(st.H.double()).mean() \
                * torch.eye(st.H.shape[0], dtype=torch.float64)
            Wqg, ginfo = gptq_quantize(W, st.H, bits)
            meas_damped = float(torch.trace(
                (W.double() - Wqg.double()).T @ (W.double() - Wqg.double()) @ Hd))
            surr.append(dict(layer=name, bits=bits,
                             surrogate=ginfo["surrogate"],
                             measured_damped=meas_damped))
    surr_err = max(abs(r["surrogate"] - r["measured_damped"]) /
                   max(r["measured_damped"], 1e-12) for r in surr)
    gptq_wins = {}
    for bits in (2, 3, 4):
        wl = [next(r for r in surr if r["layer"] == ln and r["bits"] == bits)
              for ln in layer_names]
        p0 = [next(p for p in parity if p["layer"] == ln and p["bits"] == bits
                   and p["method"] == "rtn") for ln in layer_names]
        gptq_wins[bits] = all(s["measured_damped"] <= p["measured_mse"] + 1e-9
                              for s, p in zip(wl, p0))
    print("  GPTQ surrogate identity rel-err %.2e ; GPTQ <= RTN measured: %s"
          % (surr_err, gptq_wins), flush=True)
    write_tsv("04_gptq_surrogate.tsv",
              ["layer", "bits", "surrogate", "measured_damped"],
              [(r["layer"], r["bits"], r["surrogate"], r["measured_damped"])
               for r in surr])
    write_tsv("03_parity.tsv",
              ["layer", "bits", "method", "predicted_mse", "measured_mse",
               "clip_fraction"],
              [(p["layer"], p["bits"], p["method"], p["predicted_mse"],
                p["measured_mse"], p["clip_fraction"]) for p in parity])

    # dCE vs predicted total MSE across the E2 grid
    grid = []
    for r in E2:
        bits = r["bits"]
        tot = sum(p["predicted_mse"] for p in parity
                  if p["bits"] == bits and p["method"] == r["method"])
        grid.append((tot, r["dce"]))
    xs = np.array([g[0] for g in grid]); ys = np.array([g[1] for g in grid])
    corr = float(np.corrcoef(xs, ys)[0, 1])
    slope, intercept = np.polyfit(xs, ys, 1)
    print("  dCE vs predicted MSE covariate: corr %.4f" % corr, flush=True)

    # E3b: why the isotropic prediction fails at 2 bits. Trained weight rows
    # align with the input covariance's top eigenspace (kappa > 1), and at
    # 2 bits the error direction IS the weight direction, so the directional
    # prediction v_r * mean(w_r^T H w_r / ||w_r||^2) applies instead.
    dir_rows = []
    for name, module in quantizable_linears(model):
        W = module.weight.data
        st = calib[name]
        kappa = T.alignment_factor(W, st.H)
        var, bias, cf = T.row_error_moments(W, 2, "per_channel")
        Hd = st.H.double(); Wd = W.double()
        quad = ((Wd @ Hd) * Wd).sum(dim=1)
        norms = (Wd ** 2).sum(dim=1)
        dir_pred = float(var.sum() * (quad / norms.clamp_min(1e-300)).mean())
        iso_pred = T.predicted_recon_mse(W, 2, st.H, "per_channel")["predicted_mse"]
        Wq0, _ = quantize_rows(W, 2)
        meas = T.measured_recon_mse(W, Wq0, st.H)
        dir_rows.append(dict(layer=name, kappa=kappa, isotropic=iso_pred,
                             directional=dir_pred, measured=meas,
                             iso_ratio=meas / max(iso_pred, 1e-300),
                             dir_ratio=meas / max(dir_pred, 1e-300)))
    iso_dev = max(abs(np.log(r["iso_ratio"])) for r in dir_rows)
    dir_dev = max(abs(np.log(r["dir_ratio"])) for r in dir_rows)
    kappa_mean = float(np.mean([r["kappa"] for r in dir_rows]))
    print("  2-bit alignment: kappa mean %.2f | worst |log ratio| iso %.2f dir %.2f"
          % (kappa_mean, iso_dev, dir_dev), flush=True)
    write_tsv("03b_alignment.tsv",
              ["layer", "kappa", "isotropic_pred", "directional_pred",
               "measured", "iso_ratio", "dir_ratio"],
              [(r["layer"], r["kappa"], r["isotropic"], r["directional"],
                r["measured"], r["iso_ratio"], r["dir_ratio"])
               for r in dir_rows])

    # ---- E4: water-filling allocation ----
    # Budget: 2.5 bits per weight on average -- strictly less memory than
    # uniform 3 bits, strictly more than uniform 2. Cost table: the MEASURED
    # reconstruction surrogate per layer per bit on the calibration Hessian
    # (RTN for the RTN allocation, GPTQ for the GPTQ allocation). GPTQ at
    # 2 bits is already near-lossless on this testbed, so the headroom for
    # mixed precision sits in the RTN regime; both are reported.
    layer_stats = []
    with torch.inference_mode():
        for name, module in quantizable_linears(model):
            W = module.weight.data
            layer_stats.append(dict(name=name, n_params=int(W.numel())))
    total_params = sum(s["n_params"] for s in layer_stats)

    def build_table(method):
        table = {}
        for i, (name, module) in enumerate(quantizable_linears(model)):
            W = module.weight.data
            st = calib[name]
            for b in (2, 3, 4, 5):
                if method == "rtn":
                    Wq, _ = quantize_rows(W, b)
                else:
                    Wq, _ = gptq_quantize(W, st.H, b)
                table[(i, b)] = T.measured_recon_mse(W, Wq, st.H)
        return table

    tables = {m: build_table(m) for m in ("rtn", "gptq")}
    E4 = {}
    for method in ("rtn", "gptq"):
        table = tables[method]
        budget = int(round(total_params * 2.5))
        alloc, used = T.allocate_bits(
            layer_stats, budget,
            cost_fn=lambda i, b, t=table: t[(i, b)], b_min=2, b_max=4)
        alloc_map = {layer_names[i]: alloc[i]
                     for i in range(len(layer_names))}
        qm_u2, _ = quantize_model(model, 2, method, "per_channel",
                                  calibration=calib)
        ev_u2 = eval_ce(qm_u2, tokenizer, seed=a.seed)
        qm_u3, _ = quantize_model(model, 3, method, "per_channel",
                                  calibration=calib)
        ev_u3 = eval_ce(qm_u3, tokenizer, seed=a.seed)
        qm_a, _ = quantize_model(model, None, method, "per_channel",
                                 calibration=calib, per_layer_bits=alloc_map)
        ev_a = eval_ce(qm_a, tokenizer, seed=a.seed)
        E4[method] = dict(
            budget_bits_per_weight=2.5, budget_memory=int(budget),
            used_memory=int(used),
            uniform2_dce=ev_u2["ce"] - base["ce"],
            uniform3_dce=ev_u3["ce"] - base["ce"],
            allocated_dce=ev_a["ce"] - base["ce"],
            allocation={layer_names[i]: alloc[i]
                        for i in range(len(layer_names))})
        print("  [%s] uniform-2 %+.4f | allocated(2.5bpw) %+.4f | "
              "uniform-3 %+.4f (17%% more memory)" % (
                  method, E4[method]["uniform2_dce"],
                  E4[method]["allocated_dce"], E4[method]["uniform3_dce"]),
              flush=True)
        print("  [%s] allocation: %s" % (
            method, {k.replace('.', '/'): v for k, v in alloc_map.items()}),
            flush=True)
    E4["kkt_marginal_gap_rtn"] = float(T.kkt_marginal_gap(
        layer_stats,
        {i: E4["rtn"]["allocation"][layer_names[i]]
         for i in range(len(layer_names))},
        cost_fn=lambda i, b: tables["rtn"][(i, b)]))
    write_tsv("05_allocation.tsv",
              ["layer", "n_params", "rtn_bits", "gptq_bits"],
              [(s["name"], s["n_params"],
                E4["rtn"]["allocation"][s["name"]],
                E4["gptq"]["allocation"][s["name"]])
               for s in layer_stats])

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
    bs = sorted({r["bits"] for r in E1})
    for gran, marker in (("per_tensor", "s"), ("per_channel", "o")):
        ys1 = [next(r["dce"] for r in E1 if r["bits"] == b and r["granularity"] == gran)
               for b in bs]
        ax[0].plot(bs, ys1, marker + "-", label=f"RTN {gran}")
    for method, marker, color in (("rtn", "v", "#888"), ("rtn+clip", "^", "#e76f51"),
                                  ("gptq", "d", "#2a9d8f"), ("gptq+clip", "p", "#264653")):
        rows = [r for r in E2 if r["method"] == method]
        ax[0].plot([r["bits"] for r in rows], [r["dce"] for r in rows],
                   marker + "-", color=color, label=method)
    ax[0].set_xlabel("bits"); ax[0].set_ylabel("dCE (nats/token)")
    ax[0].set_yscale("symlog", linthresh=1e-3)
    ax[0].set_title("Quantization frontier"); ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)
    for method, color in (("rtn", "#888"), ("rtn+clip", "#e76f51")):
        ps = [p for p in parity if p["method"] == method]
        ax[1].scatter([p["predicted_mse"] for p in ps],
                      [p["measured_mse"] for p in ps], s=12, alpha=.6,
                      label=method, color=color)
    lims = [min(p["predicted_mse"] for p in parity),
            max(p["predicted_mse"] for p in parity)]
    ax[1].plot(lims, lims, "k--", lw=1, label="prediction = measurement")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("predicted recon MSE (closed form)")
    ax[1].set_ylabel("measured recon MSE")
    ax[1].set_title("Theory parity per layer"); ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3, which="both")
    ax[2].bar(range(len(layer_names)),
              [E4["rtn"]["allocation"][n] for n in layer_names],
              color="#2a9d8f")
    ax[2].set_xticks(range(len(layer_names)))
    ax[2].set_xticklabels([n.replace(".", "/") for n in layer_names],
                          rotation=75, fontsize=6)
    ax[2].set_ylabel("allocated bits (RTN)")
    ax[2].set_title("Mixed precision at 2.5 bits/weight")
    ax[2].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_quant.png", dpi=140); plt.close(fig)

    summary = dict(seed=a.seed, base=base,
                   metadata=dict(parameters=model.parameter_count,
                                 n_quantized_layers=len(layer_names),
                                 vocab_size=tokenizer.vocab_size),
                   E1_rtn=E1, E2_methods=E2,
                   E3=dict(worst_logratio=worst_logratio,
                           surrogate_identity_relerr=float(surr_err),
                           gptq_leq_rtn=gptq_wins,
                           dce_vs_pred_corr=corr,
                           dce_vs_pred_slope=float(slope),
                           dce_vs_pred_intercept=float(intercept),
                           alignment=dict(kappa_mean=kappa_mean,
                                          worst_logratio_isotropic=iso_dev,
                                          worst_logratio_directional=dir_dev)),
                   E4=E4, total_seconds=time.time() - t0)
    write_json("06_summary.json", summary)
    print(f"OK {summary['total_seconds']:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
