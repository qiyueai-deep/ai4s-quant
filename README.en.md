# From-Scratch LLM Weight Quantization: Closed-Form Error Prediction, GPTQ and Water-Filling Mixed Precision

[中文](README.md) | English

This repository implements a complete quantization chain: a deterministic high-entropy corpus, a
1.82M-parameter language model trained from scratch, calibrated Hessians, two quantizers (RTN and GPTQ
at int8/4/3/2 bits), fake-quant evaluation, and a testable error theory. No `transformers`, no network
access, no GPU — every number in this README is produced by the repository code on a single CPU.

GPTQ itself is a known method (Frantar et al., OPTQ/GPTQ, ICLR 2023). This project re-implements its
core algorithm from scratch and attributes it as such. The original contribution is three theoretical
analyses around it plus an allocation algorithm, each validated empirically.

## Four propositions and measured results

**T1 Uniform rounding variance (high-bit limit)**

$$
\mathrm{Var}[\epsilon_q]=\frac{\Delta^{2}}{12}
$$

Measured: 2M-sample simulation, relative deviation < 1%.

**T2 Closed-form prediction of layer reconstruction error**

$$
E[d^{\top}Hd]=\sum_r v_r\,\mathrm{tr}(H)+\sum_r b_r^{2}\,\mathbf{1}^{\top}H\mathbf{1}
$$

With the variance and rounding-bias cross term, and truncation handled by per-row 1-D numerical
integration. Measured: worst deviation 22.6% at 3-bit, 12.1% at 4-bit; **fails at 2-bit (3–6×)**, with the
failure mechanism quantified by the alignment amplification factor κ (mean 16.7, max 34.1).

**T3 GPTQ second-order proxy identity, and GPTQ ≤ RTN**

$$
\sum_i \frac{(w_i-q_i)^{2}}{U_{ii}^{2}}=\big\lVert (W-Q)H^{1/2}\big\rVert^{2}
$$

Measured: identity relative error 6.85e-08; GPTQ ≤ RTN on all 16 layers × 3 bit widths; correlation
between total reconstruction error and ΔCE is 0.946.

**T4 Water-filling mixed precision: minimise total reconstruction error under a memory budget**

$$
\min_{\{b_\ell\}}\ \sum_\ell \mathrm{err}_\ell(b_\ell)
\quad\text{s.t.}\quad
\frac{1}{L}\sum_\ell b_\ell \le b
$$

Measured: at 2.5 bit/weight, RTN allocation reduces the uniform-2 loss from +1.158 to +0.659 (43%);
GPTQ allocation reaches −0.0000 (lossless), versus +0.0053 for uniform-2.

The most direct 2-bit result: RTN +1.158 → GPTQ +0.005, a **217× reduction in loss**.

## System

- **Corpus:** a deterministic high-entropy configuration-log language (the `hard` variant in
  `lmlab/data.py`), 9,000 lines, 154k characters, 36-token vocabulary, with wider numeric ranges and
  more line types than the `easy` variant, pushing the model to its capacity boundary. This is a key
  design decision: a model trained on a permissive corpus is too robust to weight noise, and the whole
  2–4 bit frontier shows no signal (on the `easy` corpus, per-channel 3-bit
  $\Delta\mathrm{CE} \approx 0$ and method comparison was meaningless).
- **Model:** d = 192, 4 layers, 4 heads, d_ff = 768, 1,820,544 parameters, validation CE 0.929,
  800 Adam steps in about 4 CPU minutes; benchmark CE 0.9386 (perplexity 2.56).
- **Quantization targets:** all 16 `nn.Linear` layers in the block (qkv, attention output, both MLP
  layers). `lm_head` is tied to the token embedding and is not quantized — stated explicitly.
- **Protocol:** everything is fake-quant (dequantize, then ordinary matmul forward), the standard
  post-training-quantization evaluation protocol; no int4/int8 kernels are included.

## Evaluation results

### E1: RTN frontier (bits × granularity)

$\Delta\mathrm{CE}$ is the cross-entropy increase, in nats/token.

| bits | per-tensor | per-channel |
|---|---|---|
| 2 | +6.427 (ppl 1581) | +1.158 |
| 3 | +0.289 | +0.019 |
| 4 | +0.012 | +0.0015 |
| 8 | +0.0001 | +0.0001 |

### E2: Method comparison (per-channel)

| bits | RTN | RTN+clip | GPTQ | GPTQ+clip |
|---|---|---|---|---|
| 2 | +1.158 | +0.064 | **+0.005** | +0.015 |
| 3 | +0.019 | +0.008 | **−0.000** | −0.000 |
| 4 | +0.0015 | +0.0005 | +0.0001 | +0.0005 |

Two counter-intuitive results are reported as measured: an MSE-optimal clipping search improves RTN by
18× at 2-bit (+1.158 → +0.064), yet combined with GPTQ it is slightly worse (+0.015 vs +0.005) — the
two error-redistribution mechanisms overlap, and clipping changes the weight distribution shape GPTQ
relies on.

### E3: Theory versus measurement

- The closed-form prediction (T2) deviates at worst by 22.6% / 12.1% at 3/4-bit; all 96 per-layer
  measurement points map one-to-one to `results_quant/03_parity.tsv`.
- **2-bit failure and κ:** the isotropic prediction assumes rounding-error directions are random, but at
  2-bit each weight takes only $\{-1, 0, 1\}\times\Delta$, small-magnitude weights round to zero and the error equals
  the weight itself, concentrating error in the row's small-coordinate subspace. Trained weight rows are
  simultaneously aligned with the top eigenspace of the input covariance, which defines the alignment
  amplification factor:

  $$
  \kappa=\frac{w^{\top}Hw}{\lVert w\rVert^{2}\cdot\mathrm{tr}(H)/d_{\mathrm{in}}}
  $$

  It has mean 16.7 and max 34.1 over 16 layers, while trace(H) averages over all feature directions —
  hence the systematic 3–6× underestimation. A directional prediction (error ∝ weight direction) is also
  wrong, since the true error direction is the row's small-coordinate subspace. Conclusion: there is no
  cheap closed-form prediction at 2-bit, so the mixed-precision allocator uses a measured error table on a
  calibration set — a methodological choice, made explicit.
- The GPTQ proxy identity holds to 6.85e-08 relative error, a strong correctness assertion for the
  implementation.

### E4: Water-filling mixed precision

Budget 2.5 bit/weight (17% less memory than uniform-3, 0.5 more than uniform-2):

| Method | uniform-2 | allocation (2.5 bpw) | uniform-3 |
|---|---|---|---|
| RTN | +1.158 | **+0.659** | +0.019 |
| GPTQ | +0.0053 | **−0.0000** | −0.0002 |

The allocator combines a greedy marginal step with swap-based local search (pure greedy leaves unused
memory fragments and measured 9% worse than brute force; swap search matched the brute-force optimum in
small test instances). Cost tables come from measured reconstruction error on the calibration set. The
RTN allocation promotes attention output layers to 4-bit and compresses MLP layers to 2-bit; GPTQ
allocation is lossless at the same budget.

## Reproducing

```bash
pip install -r requirements.txt
python tests_quant.py -v                                       # 17 assertions
python -m lmlab.train --force --corpus hard --d-model 192 --n-layers 4 --suffix q
python experiments_quant.py                                    # about 1 CPU minute
python experiments_quant.py --seed 7 --out results_quant_seed7 # alternative seed
```

The whole chain is deterministic: corpus, initialisation, calibration window and evaluation window all
use fixed seeds, and reruns are byte-identical (training takes about 4 minutes once; later experiments
read the checkpoint directly).

## Explicit boundaries

- Character-level 36-token vocabulary, synthetic corpus, 1.82M parameters: the theoretical structure and
  conclusions transfer, but the absolute numbers do not correspond to 7B-scale models.
- There is no exact closed-form prediction at 2-bit: the failure mechanism (small-coordinate dominance
  plus κ alignment amplification) is quantified, but accurate prediction requires measurement on a
  calibration set. This is a conclusion, not a defect to be hidden.
- Closed-form prediction for RTN+clip is mediocre (0.5–0.6 log-deviation): after clipping, the error
  distribution departs from the Gaussian model and no finer modelling is attempted.
- Fake-quant protocol: no real integer kernels, and no measured memory or bandwidth figures.
- KKT marginal ratios are descriptive only when using measured cost tables (which are non-smooth);
  allocator optimality is validated against brute force on small instances.

## Directory

```
lmlab/               corpus, model, training (infrastructure)
quant/               calibration, RTN, GPTQ, theory, pipeline, evaluation
experiments_quant.py E1-E4 experiments
tests_quant.py       17 assertions
artifacts/           self-trained checkpoints, tokenizer, manifest
data/                deterministic corpus (regenerable)
results_quant/       tables, figures, summary.json
_legacy_specdec/     archived speculative-decoding project
theory_lab/          archived statistical-physics project
```
