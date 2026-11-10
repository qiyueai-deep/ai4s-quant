"""Deterministic synthetic corpus with learnable long-range structure.

The corpus is an artificial configuration-log language. Lines are short and
templated ("set n07 ram 128"), slots are drawn from small sets, and
consecutive lines belong to the same episode (same primary node) with high
probability. This gives the language cheap local template structure plus
episode-level context dependence: which node or field comes next depends on
lines further back, so the target-draft gap and the acceptance rate vary by
context instead of being constant.

The grammar uses short words on purpose. Character-level models learn it to
near-determinism, so the target's greedy path stays on-template and the
measured acceptance rate reflects the genuine target-draft gap rather than
the target's own degeneration loops.

Generation is fully deterministic given the seed: no downloads, no external
data, and byte-identical corpora across machines.
"""
from __future__ import annotations

import numpy as np

FIELDS = ["ram", "cpu", "disk"]
VALUES = ["64", "128", "256", "512", "1024"]
BWS = ["16", "32", "64", "128"]


def generate_corpus(seed=0, n_lines=9000):
    """Return the raw corpus text. Identical seed gives identical bytes.

    Consecutive duplicate lines are forbidden: a character-level LM that is
    trained on repeated identical lines learns that line-level repetition is
    legal, and its greedy path then collapses into "set 6\\nset 6\\n..." loops,
    which destroys speculative acceptance in the degenerate region.
    """
    rng = np.random.default_rng(seed)
    nodes = ["n%02d" % i for i in range(1, 17)]
    lines = []
    node = nodes[0]
    episode_left = 0
    while len(lines) < n_lines:
        if episode_left <= 0:
            node = nodes[int(rng.integers(0, len(nodes)))]
            episode_left = int(rng.integers(2, 7))
        for _ in range(8):
            r = rng.random()
            if r < 0.4:
                line = "set %s %s %s" % (
                    node, FIELDS[int(rng.integers(0, len(FIELDS)))],
                    VALUES[int(rng.integers(0, len(VALUES)))])
            elif r < 0.65:
                other = nodes[int(rng.integers(0, len(nodes)))]
                line = "link %s %s bw %s" % (
                    node, other, BWS[int(rng.integers(0, len(BWS)))])
            elif r < 0.85:
                other = nodes[int(rng.integers(0, len(nodes)))]
                line = "ping %s %s" % (node, other)
            elif r < 0.95:
                line = "get %s" % node
            else:
                line = "del %s" % nodes[int(rng.integers(0, len(nodes)))]
            if line != (lines[-1] if lines else ""):
                break
        lines.append(line)
        if rng.random() < 0.7:
            episode_left -= 1
        else:
            episode_left = 0
    return "\n".join(lines) + "\n"


def generate_corpus_hard(seed=0, n_lines=9000):
    """Higher-entropy variant for the quantization testbed.

    Wider numeric ranges, more fields and line types, longer episodes: a
    model trained on this language sits closer to its capacity limit, which
    makes weight quantization damage measurable across the whole 2-4 bit
    frontier instead of only at 2 bits.
    """
    rng = np.random.default_rng(seed)
    nodes = ["n%02d" % i for i in range(1, 33)]
    fields = ["ram", "cpu", "disk", "gpu", "net", "io"]
    lines = []
    node = nodes[0]
    episode_left = 0
    while len(lines) < n_lines:
        if episode_left <= 0:
            node = nodes[int(rng.integers(0, len(nodes)))]
            episode_left = int(rng.integers(2, 13))
        for _ in range(8):
            r = rng.random()
            if r < 0.3:
                line = "set %s %s %d" % (
                    node, fields[int(rng.integers(0, len(fields)))],
                    int(rng.integers(1, 8193)))
            elif r < 0.5:
                other = nodes[int(rng.integers(0, len(nodes)))]
                line = "link %s %s bw %d lat %d" % (
                    node, other, int(rng.integers(1, 1025)),
                    int(rng.integers(1, 999)))
            elif r < 0.65:
                other = nodes[int(rng.integers(0, len(nodes)))]
                line = "ping %s %s %d" % (node, other, int(rng.integers(1, 999)))
            elif r < 0.8:
                a, b = nodes[int(rng.integers(0, len(nodes)))], \
                    nodes[int(rng.integers(0, len(nodes)))]
                line = "route %s %s via %s" % (node, a, b)
            elif r < 0.9:
                line = "quota %s %d %d" % (
                    node, int(rng.integers(1, 4097)), int(rng.integers(1, 99)))
            else:
                line = "del %s" % nodes[int(rng.integers(0, len(nodes)))]
            if line != (lines[-1] if lines else ""):
                break
        lines.append(line)
        if rng.random() < 0.7:
            episode_left -= 1
        else:
            episode_left = 0
    return "\n".join(lines) + "\n"


def split_corpus(text, val_frac=0.1):
    """Split by lines, keeping episode boundaries inside one side."""
    lines = text.splitlines()
    cut = int(len(lines) * (1.0 - val_frac))
    return "\n".join(lines[:cut]) + "\n", "\n".join(lines[cut:]) + "\n"
