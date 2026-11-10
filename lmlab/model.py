"""Decoder-only Transformer, tokenizer and checkpoint I/O for this project.

The architecture is a compact GPT-style decoder with tied embeddings and an
incremental KV cache used by the speculative decoding engine. Training,
distillation and evaluation all happen inside this repository; the
checkpoints under artifacts/ are produced by lmlab.train and loaded here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384
    dropout: float = 0.0


@dataclass
class KVCache:
    n_layers: int
    keys: list = field(init=False)
    values: list = field(init=False)

    def __post_init__(self):
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers

    @property
    def length(self):
        first = self.keys[0]
        return 0 if first is None else int(first.shape[2])

    def get(self, layer):
        return self.keys[layer], self.values[layer]

    def append(self, layer, key, value):
        old_key, old_value = self.get(layer)
        self.keys[layer] = key if old_key is None else torch.cat([old_key, key], dim=2)
        self.values[layer] = value if old_value is None else torch.cat([old_value, value], dim=2)

    def truncate(self, new_length):
        for layer in range(self.n_layers):
            if self.keys[layer] is not None:
                self.keys[layer] = self.keys[layer][:, :, :new_length, :].contiguous()
                self.values[layer] = self.values[layer][:, :, :new_length, :].contiguous()


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, cache, layer_index):
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        past_key, past_value = (None, None) if cache is None else cache.get(layer_index)
        past_len = 0 if past_key is None else past_key.shape[2]
        all_key = key if past_key is None else torch.cat([past_key, key], dim=2)
        all_value = value if past_value is None else torch.cat([past_value, value], dim=2)
        scores = query @ all_key.transpose(-2, -1) / math.sqrt(self.head_dim)
        qpos = past_len + torch.arange(seq_len, device=x.device)
        kpos = torch.arange(past_len + seq_len, device=x.device)
        mask = kpos.unsqueeze(0) > qpos.unsqueeze(1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        output = (attn @ all_value).transpose(1, 2).contiguous().view(batch, seq_len, width)
        if cache is not None:
            cache.append(layer_index, key, value)
        return self.out(output)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff), nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, x, cache, layer_index):
        x = x + self.attention(self.norm1(x), cache, layer_index)
        return x + self.mlp(self.norm2(x))


class TinyTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def new_cache(self):
        return KVCache(self.config.n_layers)

    def forward(self, input_ids, cache=None):
        past_len = 0 if cache is None else cache.length
        if past_len + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        positions = torch.arange(past_len, past_len + input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        for index, block in enumerate(self.blocks):
            x = block(x, cache, index)
        return self.lm_head(self.final_norm(x))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class CharTokenizer:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["tokens"])

    @property
    def vocab_size(self):
        return len(self.tokens)

    @property
    def bos_id(self):
        return self.token_to_id[self.BOS]

    def encode(self, text, add_bos=False):
        ids = [self.token_to_id.get(c, self.token_to_id[self.UNK]) for c in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        return ids

    def decode(self, ids, skip_special=True):
        special = {self.PAD, self.BOS, self.EOS, self.UNK}
        return "".join("" if skip_special and self.tokens[int(i)] in special
                       else self.tokens[int(i)] for i in ids)


def _load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    model = TinyTransformerLM(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_artifacts(artifact_dir):
    artifact_dir = Path(artifact_dir)
    target = _load_checkpoint(artifact_dir / "target.pt")
    draft = _load_checkpoint(artifact_dir / "draft.pt")
    tokenizer = CharTokenizer.load(artifact_dir / "tokenizer.json")
    return target, draft, tokenizer


def load_checkpoint(path):
    """Public wrapper around the checkpoint loader."""
    return _load_checkpoint(path)
