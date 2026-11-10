"""Single-target training for the quantization testbed.

    python -m lmlab.train --force                    # easy corpus
    python -m lmlab.train --force --corpus hard      # harder corpus
    python -m lmlab.train --force --corpus hard \
        --d-model 192 --n-layers 4 --suffix q        # the quantization model

Deterministic: fixed seeds for corpus, initialisation and batch sampling.
Saves artifacts/target{suffix}.pt, tokenizer{suffix}.json, manifest{suffix}.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import generate_corpus, generate_corpus_hard, split_corpus
from .model import CharTokenizer, ModelConfig, TinyTransformerLM

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ROOT / "data"


def init_weights(model):
    """GPT-style small-variance init; the default N(0,1) embedding blows up
    the tied lm_head logits and slows convergence."""
    for emb in (model.token_embedding, model.position_embedding):
        nn.init.normal_(emb.weight, std=0.02)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)


def sample_batch(ids, batch, seq_len, rng):
    starts = rng.integers(0, len(ids) - seq_len - 1, size=batch)
    x = np.stack([ids[s:s + seq_len + 1] for s in starts])
    return torch.from_numpy(x)


@torch.no_grad()
def eval_ce(model, ids, n_batches=8, batch=32, seq_len=48, rng=None):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x = sample_batch(ids, batch, seq_len, rng)
        logits = model(x[:, :-1])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def train_target(tokenizer, train_ids, val_ids, steps, batch, seq_len,
                 lr, seed, d_model=192, n_layers=4, d_ff=768, log_every=50):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=192,
                      d_model=d_model, n_layers=n_layers, n_heads=4, d_ff=d_ff)
    model = TinyTransformerLM(cfg)
    init_weights(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=lr * 0.1)
    rng = np.random.default_rng(seed + 1)
    model.train()
    for step in range(1, steps + 1):
        x = sample_batch(train_ids, batch, seq_len, rng)
        logits = model(x[:, :-1])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == 1:
            print(f"  step {step:4d}/{steps}  loss {float(loss.detach()):.4f}",
                  flush=True)
    val = eval_ce(model, val_ids, rng=rng)
    print(f"  val CE {val:.4f}", flush=True)
    return model, val


def save_checkpoint(path, model):
    torch.save({"config": dict(model.config.__dict__),
                "state_dict": model.state_dict()}, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--corpus", choices=("easy", "hard"), default="hard")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=768)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--suffix", type=str, default="q")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    ART.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    marker = ART / f"manifest{a.suffix}.json"
    if marker.exists() and not a.force:
        print("artifacts already present, skip training (use --force)")
        return

    t0 = time.time()
    corpus = (generate_corpus_hard if a.corpus == "hard" else generate_corpus)(
        seed=a.seed)
    train_text, val_text = split_corpus(corpus)
    (DATA / f"corpus_train_{a.suffix}.txt").write_text(train_text, encoding="utf-8")
    (DATA / f"corpus_val_{a.suffix}.txt").write_text(val_text, encoding="utf-8")
    chars = sorted(set(train_text))
    specials = [CharTokenizer.PAD, CharTokenizer.BOS, CharTokenizer.EOS,
                CharTokenizer.UNK]
    tokenizer = CharTokenizer(specials + [c for c in chars if c not in specials])
    (ART / f"tokenizer{a.suffix}.json").write_text(
        json.dumps({"tokens": tokenizer.tokens}), encoding="utf-8")
    print(f"corpus {a.corpus}: train {len(train_text)} chars, "
          f"vocab {tokenizer.vocab_size}", flush=True)

    def encode(text):
        return np.array(tokenizer.encode(text, add_bos=True), dtype=np.int64)

    train_ids, val_ids = encode(train_text), encode(val_text)
    model, val_ce = train_target(tokenizer, train_ids, val_ids, a.steps,
                                 a.batch, a.seq_len, lr=4e-3, seed=a.seed,
                                 d_model=a.d_model, n_layers=a.n_layers,
                                 d_ff=a.d_ff)
    save_checkpoint(ART / f"target{a.suffix}.pt", model)
    manifest = dict(
        seed=a.seed, corpus=a.corpus,
        config=dict(model.config.__dict__),
        parameters=model.parameter_count,
        vocab_size=tokenizer.vocab_size,
        corpus_train_chars=len(train_text), corpus_val_chars=len(val_text),
        steps=a.steps, val_ce=val_ce, train_seconds=time.time() - t0,
    )
    marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str)[:600])
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
