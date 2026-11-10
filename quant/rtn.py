"""Symmetric round-to-nearest weight quantization (fake-quant evaluation).

A symmetric uniform quantizer with B bits has levels {-L, ..., L},
L = 2^(B-1) - 1, and step size Delta = s where s is the scale. The rounding
error is bounded by Delta/2 and, away from clipping, has variance Delta^2/12
-- the quantity the closed-form theory predictions are built on.
"""
from __future__ import annotations

import torch


def levels(bits):
    return 2 ** (bits - 1) - 1


def quantize_rows(W, bits):
    """Per-output-channel (row-wise) symmetric RTN. Returns (Wq, scales)."""
    L = float(levels(bits))
    scales = W.abs().amax(dim=1, keepdim=True) / L          # (out, 1)
    scales = torch.clamp(scales, min=1e-12)
    q = torch.round(W / scales)
    q = torch.clamp(q, -L, L)
    return q * scales, scales.squeeze(1)


def quantize_tensor(W, bits):
    """Per-tensor symmetric RTN. Returns (Wq, scale)."""
    L = float(levels(bits))
    scale = W.abs().max() / L
    scale = torch.clamp(scale, min=1e-12)
    q = torch.round(W / scale)
    q = torch.clamp(q, -L, L)
    return q * scale, scale.reshape(())


def quantize(W, bits, granularity):
    """granularity: 'per_channel' (row-wise) or 'per_tensor'."""
    if granularity == "per_channel":
        return quantize_rows(W, bits)
    if granularity == "per_tensor":
        return quantize_tensor(W, bits)
    raise ValueError(granularity)


def max_abs_error(W, Wq):
    return float((W - Wq).abs().max())


CLIP_CANDIDATES = (100.0, 99.9, 99.0, 97.5, 95.0, 90.0)


def quantize_rows_clip(W, bits, H=None):
    """Per-row symmetric RTN with MSE-optimal clipping.

    For each candidate clip percentile p the row scale is
    s_r = quantile(|w_r|, p) / L, weights above s_r are clipped, and the row
    objective is the calibration-weighted reconstruction error
    d_r^T (X^T X) d_r with d_r = w_r - q_r. The per-row problems are
    independent, so the search is exact per row; H = 2 X^T X comes from the
    calibration pass, or an identity H (plain MSE) is used when it is None.

    Returns (Wq, scales, chosen_percentile_per_row).
    """
    out_f, in_f = W.shape
    L = float(levels(bits))
    Wd = W.double()
    absW = Wd.abs()
    Hf = (torch.eye(in_f, dtype=torch.float64) if H is None
          else H.double() / 2.0)
    best = None
    best_err = torch.full((out_f,), float("inf"), dtype=torch.float64)
    best_scales = torch.ones(out_f, dtype=torch.float64)
    best_pct = torch.full((out_f,), 100.0)
    for p in CLIP_CANDIDATES:
        s = torch.quantile(absW, p / 100.0, dim=1) / L
        s = torch.clamp(s, min=1e-12)
        q = torch.clamp(torch.round(Wd / s.unsqueeze(1)), -L, L) * s.unsqueeze(1)
        d = Wd - q
        # per-row surrogate: diag(D H D^T)
        err = torch.einsum("ij,jk,ik->i", d, Hf, d)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scales = torch.where(better, s, best_scales)
        best_pct = torch.where(better, torch.full_like(s, p), best_pct)
    q = torch.clamp(torch.round(Wd / best_scales.unsqueeze(1)), -L, L) \
        * best_scales.unsqueeze(1)
    return q.float(), best_scales.float(), best_pct
