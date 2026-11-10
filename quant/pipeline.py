"""Fake-quantization pipeline and perplexity evaluation.

`quantize_model` returns a model COPY whose block-Linear weights are replaced
by their dequantized versions (fake quantization, the standard PTQ
evaluation protocol). The lm_head is tied to the token embedding and is
excluded. With clip=True the per-row scales come from an MSE-optimal
clipping search on the calibration Hessian instead of plain max-abs.
`eval_ce` measures teacher-forced cross-entropy and perplexity on
deterministic validation windows; it is fully reproducible.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .calibration import quantizable_linears
from .gptq import gptq_quantize
from .rtn import quantize, quantize_rows_clip


def quantize_model(model, bits, method="rtn", granularity="per_channel",
                   calibration=None, per_layer_bits=None, damp_frac=0.01,
                   clip=False, suffix="q"):
    """Return (quantized copy, layer_info list).

    method: 'rtn' or 'gptq'. per_layer_bits: optional dict layer_name -> bits
    overriding the global bits (used by the water-filling allocation).
    clip: MSE-optimal per-row clipping search (needs calibration stats).
    calibration: {layer_name: LayerStats}, required for gptq and clip.
    """
    qmodel = copy.deepcopy(model)
    qmodel.eval()
    info = []
    targets = dict(quantizable_linears(qmodel))
    calib = calibration or {}
    with torch.inference_mode():
        for name, module in targets.items():
            W = module.weight.data
            b = per_layer_bits.get(name, bits) if per_layer_bits else bits
            st = calib.get(name)
            scales = None
            if clip:
                if st is None:
                    raise KeyError(f"clip needs calibration stats for {name}")
                Wq, scales, pct = quantize_rows_clip(W, b, st.H)
            if method == "gptq":
                if st is None:
                    raise KeyError(f"gptq needs calibration stats for {name}")
                Wq, extra = gptq_quantize(W, st.H, b, damp_frac=damp_frac,
                                          scales=scales)
            elif method == "rtn":
                if clip:
                    extra = {}
                else:
                    Wq, _ = quantize(W, b, granularity)
                    extra = {}
            else:
                raise ValueError(method)
            module.weight.data = Wq
            info.append(dict(name=name, bits=b, shape=list(W.shape),
                             n_params=int(W.numel()), **extra))
    return qmodel, info


@torch.inference_mode()
def eval_ce(model, tokenizer, n_windows=64, seq_len=48, seed=0, suffix="q"):
    """Deterministic teacher-forced CE (nats/token) on validation windows."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    val_text = (root / "data" / f"corpus_val_{suffix}.txt").read_text(
        encoding="utf-8")
    ids = np.array(tokenizer.encode(val_text, add_bos=True), dtype=np.int64)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_windows):
        s = int(rng.integers(0, len(ids) - seq_len - 1))
        x = torch.from_numpy(ids[s:s + seq_len + 1]).unsqueeze(0)
        logits = model(x[:, :-1])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        losses.append(float(loss))
    ce = float(np.mean(losses))
    return dict(ce=ce, perplexity=float(np.exp(ce)),
                ce_std=float(np.std(losses)), n_windows=n_windows)
