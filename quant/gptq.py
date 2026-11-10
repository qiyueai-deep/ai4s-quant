"""GPTQ: closed-form column-wise rounding with second-order error feedback.

Re-implementation from scratch of the core GPTQ algorithm (Frantar et al.,
"OPTQ: GPTQ", ICLR 2023). The layer objective is the second-order surrogate
  min_q  || (W - q) H^{1/2} ||_F^2 ,   H = 2 X^T X,
which upper-bounds the layer reconstruction error E||(W-q)x||^2. With the
already-quantized columns fixed, the optimal update of the remaining columns
is closed-form, and the Cholesky factor U of H^{-1} (H^{-1} = U U^T, U upper
triangular) turns the column loop into:

    for i in columns:
        w   = W[:, i] + feedback[:, i]
        q_i = RTN(w)                       # per-row symmetric scale
        feedback[:, i+1:] += (w - q_i) / U[i, i] * U[i, i+1:]

Columns are processed in the order given by U (a fixed permutation in this
implementation: the natural order), and the surrogate decrease of each step
is exactly (w - q_i)^2 / U[i, i]^2 summed over rows, which the tests verify.

The per-row scales are computed from the ORIGINAL weights, before any
feedback, so the quantizer grid is the same as RTN's and the only difference
between the two methods is the error feedback.
"""
from __future__ import annotations

import torch

from .rtn import levels, quantize_rows


def gptq_quantize(W, H, bits, damp_frac=0.01, scales=None):
    """Quantize W (out, in) in float64 internals. Returns (Wq float32, info).

    scales: optional per-row step sizes; by default they come from plain RTN
    on the original weights. Pass the scales from the clip search to combine
    GPTQ's error feedback with MSE-optimal clipping.
    info contains the measured reconstruction surrogate on the calibration
    Hessian and the dampening fraction.
    """
    out_f, in_f = W.shape
    Wd = W.double()
    Hd = H.double().clone()
    Hd += damp_frac * torch.diag(Hd).mean() * torch.eye(in_f, dtype=torch.float64)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hd))
    U = torch.linalg.cholesky(Hinv, upper=True)     # Hinv = U U^T

    if scales is None:
        _, scales = quantize_rows(Wd.float(), bits)
    scales = scales.double()                        # (out,)
    L = float(levels(bits))

    Wq = torch.zeros_like(Wd)
    feedback = torch.zeros_like(Wd)
    surrogate = 0.0
    for i in range(in_f):
        w = Wd[:, i] + feedback[:, i]
        q = torch.clamp(torch.round(w / scales), -L, L) * scales
        Wq[:, i] = q
        d = U[i, i]
        resid = w - q
        surrogate += float((resid / d).pow(2).sum())
        if i + 1 < in_f:
            # official GPTQ update: W_remaining -= (w - q)/d * U[i, i+1:]
            feedback[:, i + 1:] -= (resid / d).unsqueeze(1) \
                * U[i, i + 1:].unsqueeze(0)
    info = dict(surrogate=surrogate, damp_frac=damp_frac)
    return Wq.float(), info


def recon_mse(W, Wq, H):
    """Measured layer reconstruction surrogate ||(W - Wq) H^{1/2}||_F^2
    computed without a square root: trace((W-Wq)^T (W-Wq) H)."""
    D = (W.double() - Wq.double())
    return float(torch.trace(D.T @ D @ H.double()))
