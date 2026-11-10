"""Calibration statistics for post-training quantization.

For every quantizable nn.Linear we collect, over a calibration pass on the
training distribution:
  H  = 2 * sum_tokens x x^T          (in x in)  GPTQ / second-order Hessian
  n  = number of tokens
  trace(H)/2 = ||X||_F^2             normalisation for reconstruction MSE

The layer reconstruction error of a weight perturbation dW satisfies, to
second order and in expectation over inputs,
  E ||dW x||^2 = trace(H * E[dW^T dW]) / 2,
so everything needed to PREDICT the reconstruction cost of a quantization
scheme is contained in H. No activations are stored beyond this summary,
which keeps the whole pipeline CPU-trivial.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lmlab.model import TinyTransformerLM


@dataclass
class LayerStats:
    name: str
    H: torch.Tensor          # 2 * sum x x^T, (in, in), float64
    n_tokens: int

    @property
    def frob_x2(self):
        """||X||_F^2 = trace(H) / 2."""
        return float(torch.trace(self.H) / 2.0)


def quantizable_linears(model: TinyTransformerLM):
    """All Linear layers inside the blocks. lm_head is tied to the token
    embedding and is therefore excluded from quantization."""
    out = []
    for name, module in model.blocks.named_modules():
        if isinstance(module, nn.Linear):
            out.append((name, module))
    return out


@torch.inference_mode()
def collect_calibration(model, tokenizer, n_windows=64, seq_len=48, seed=0,
                        suffix="q"):
    """Run the model on deterministic calibration windows and accumulate the
    per-layer Hessian summaries. Returns {layer_name: LayerStats}."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    root = Path(__file__).resolve().parents[1]
    train_text = (root / "data" / f"corpus_train_{suffix}.txt").read_text(
        encoding="utf-8")
    ids = np.array(tokenizer.encode(train_text, add_bos=True), dtype=np.int64)
    stats = {name: None for name, _ in quantizable_linears(model)}
    handles = []

    def make_hook(name):
        def hook(module, args):
            x = args[0].reshape(-1, args[0].shape[-1]).double()
            h = 2.0 * (x.T @ x)
            if stats[name] is None:
                stats[name] = LayerStats(name=name, H=h, n_tokens=x.shape[0])
            else:
                stats[name].H += h
                stats[name].n_tokens += x.shape[0]
        return hook

    for name, module in quantizable_linears(model):
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        for _ in range(n_windows):
            s = int(rng.integers(0, len(ids) - seq_len - 1))
            x = torch.from_numpy(ids[s:s + seq_len]).unsqueeze(0)
            model(x)
    finally:
        for h in handles:
            h.remove()
    return stats
