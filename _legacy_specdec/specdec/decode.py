"""KV-cache decoding engines: greedy baseline, speculative decoding, and a
theory-guided adaptive controller for the draft length.

Correctness contract: speculative output is EXACTLY the target's greedy
output, token for token. The controllers only change how many tokens are
proposed per verification round, never which tokens are emitted.
"""
from __future__ import annotations

import time

import numpy as np
import torch


@torch.inference_mode()
def greedy_generate(target, prompt_ids, n_new):
    """Pure autoregressive greedy decoding with a KV cache. Reference output."""
    cache = target.new_cache()
    ids = list(prompt_ids)
    x = torch.tensor([prompt_ids], dtype=torch.long)
    logits = target(x, cache=cache)[0, -1]
    for _ in range(n_new):
        tok = int(logits.argmax())
        ids.append(tok)
        logits = target(torch.tensor([[tok]], dtype=torch.long), cache=cache)[0, -1]
    return ids


@torch.inference_mode()
def _entropy(logits):
    p = torch.softmax(logits.double(), dim=-1)
    return float(-(p * p.clamp_min(1e-30).log()).sum())


class AdaptiveController:
    """Runtime draft-length selection from measured costs and online alpha.

    The controller needs no offline sweep. At construction it directly times
    the two primitive costs on the actual machine:
      c_block(K)  one target verification forward over K proposal tokens,
                  in units of one single-token target step,
      g           one draft step, in the same units.
    During decoding it maintains an exponentially decayed, Laplace-smoothed
    estimate of the greedy acceptance rate (fraction of proposals accepted)
    and picks
        K = argmax_K  (1 - a^(K+1)) / (1 - a) / (c_block(K) + g K),
    which is exactly the expected-tokens-over-cost model of proposition B,
    with the acceptance rate estimated online instead of assumed.

    With entropy_bucketing=True it keeps one acceptance estimate per draft
    entropy bucket, because the target-draft gap depends on the context:
    flat draft distributions are less aligned with the target than peaked
    ones. Each bucket then gets its own K.
    """

    def __init__(self, target, draft, k_grid=range(1, 33), ewma=0.95,
                 entropy_bucketing=False, entropy_threshold=None,
                 timing_repeats=30):
        self.k_grid = list(k_grid)
        self.ewma = ewma
        self.entropy_bucketing = entropy_bucketing
        self.entropy_threshold = entropy_threshold
        one = torch.tensor([[0]], dtype=torch.long)
        with torch.inference_mode():
            for _ in range(10):
                target(one)
            t0 = time.perf_counter()
            for _ in range(timing_repeats):
                target(one)
            self.target_step = (time.perf_counter() - t0) / timing_repeats
            for _ in range(10):
                draft(one)
            t0 = time.perf_counter()
            for _ in range(timing_repeats):
                draft(one)
            self.draft_step = (time.perf_counter() - t0) / timing_repeats
            self.c_block = {}
            for k in self.k_grid:
                blk = torch.full((1, k + 1), 0, dtype=torch.long)
                for _ in range(5):
                    target(blk)
                t0 = time.perf_counter()
                for _ in range(timing_repeats):
                    target(blk)
                self.c_block[k] = (time.perf_counter() - t0) / timing_repeats / self.target_step
        self.g = self.draft_step / self.target_step
        # exponentially decayed Beta counts (Laplace-smoothed acceptance rate),
        # one pair per entropy bucket. Counts track TESTED slots only: each
        # round contributes `match` successes and -- if it ended in a
        # mismatch -- exactly one failure. Counting (proposed - accepted) as
        # failures would drag the estimate down, because slots never reached
        # before the round ended are not Bernoulli failures.
        self.counts = {"low": [2.0, 0.6], "high": [2.0, 0.6]}
        self.n_rounds = {"low": 0, "high": 0}
        self.chosen = []
        if entropy_bucketing and entropy_threshold is None:
            self.entropy_threshold = self._calibrate_entropy(target, draft)

    @torch.inference_mode()
    def _calibrate_entropy(self, target, draft, steps=24):
        """Split buckets at the median DRAFT entropy seen on a greedy rollout.

        The runtime controller receives the draft's entropy, so the threshold
        must be calibrated on the same quantity; the target's own entropy has
        a completely different scale and would misroute every context.
        """
        ents = []
        ct, cd = target.new_cache(), draft.new_cache()
        x = torch.tensor([[0]], dtype=torch.long)
        lt = target(x, cache=ct)[0, -1]
        ld = draft(x, cache=cd)[0, -1]
        for _ in range(steps):
            ents.append(_entropy(ld))
            tok = int(lt.argmax())
            t = torch.tensor([[tok]], dtype=torch.long)
            lt = target(t, cache=ct)[0, -1]
            ld = draft(t, cache=cd)[0, -1]
        return float(np.median(ents))

    @property
    def acc(self):
        """Smoothed per-bucket acceptance estimates."""
        return {b: (a + 1.0) / (a + c + 2.0)
                for b, (a, c) in self.counts.items()}

    def bucket(self, draft_entropy):
        if not self.entropy_bucketing:
            return "low"
        return "low" if draft_entropy < self.entropy_threshold else "high"

    def choose_k(self, draft_entropy):
        b = self.bucket(draft_entropy)
        a = float(np.clip(self.acc[b], 0.0, 0.999))
        best_k, best_s = 1, -1.0
        for k in self.k_grid:
            # one round = one (K+1)-token verification forward + K+1 draft
            # forwards (K proposals + feeding the previous bonus)
            e = (1.0 - a ** (k + 1)) / (1.0 - a) if a < 1 else k + 1
            s = e / (self.c_block[k] + self.g * (k + 1))
            if s > best_s:
                best_k, best_s = k, s
        self.chosen.append(best_k)
        return best_k

    def update(self, match, k, draft_entropy):
        b = self.bucket(draft_entropy)
        lam = self.ewma
        self.counts[b][0] = lam * self.counts[b][0] + match
        self.counts[b][1] = lam * self.counts[b][1] + (1.0 if match < k else 0.0)
        self.n_rounds[b] += 1

    def snapshot(self):
        acc = self.acc
        return dict(acc_low=acc["low"], acc_high=acc["high"],
                    rounds_low=self.n_rounds["low"], rounds_high=self.n_rounds["high"],
                    g=self.g, target_step_seconds=self.target_step,
                    draft_step_seconds=self.draft_step,
                    entropy_threshold=self.entropy_threshold,
                    c_block={k: v for k, v in self.c_block.items()})


@torch.inference_mode()
def speculative_generate(target, draft, prompt_ids, n_new, k=None,
                         controller=None):
    """Greedy speculative decoding with exact-output guarantee.

    Canonical round structure (one target forward per round):

    1. the draft proposes K tokens autoregressively from its cached state;
    2. the target verifies them in ONE forward whose input is the previous
       round's bonus token followed by the K proposals -- the bonus token's
       key/value is computed here, at its correct position, so no separate
       feed-forward is ever needed;
    3. proposals are accepted up to the first mismatch with the target's own
       argmax; the target's argmax at the first rejected slot is emitted as
       the round's bonus and becomes the first input of the next verification.

    Caches are truncated back to the committed context after each round. The
    controllers only change K, never which tokens are emitted, so the output
    is exactly the target's greedy output.
    """
    if (controller is None) == (k is None):
        raise ValueError("pass exactly one of k= or controller=")
    ct, cd = target.new_cache(), draft.new_cache()
    out = []
    # draft processes the prompt once; the target sees the prompt inside the
    # first verification forward (prefill merged into round 1)
    x = torch.tensor([prompt_ids], dtype=torch.long)
    logits_d = draft(x, cache=cd)[0, -1]
    bonus_prev = None          # last emitted token, target KV not yet computed
    committed = len(prompt_ids)  # tokens fully covered by the target cache
    stats = dict(rounds=0, target_forwards=0, draft_forwards=1,
                 accepted=0, proposed=0, k_history=[], entropy_history=[],
                 match_history=[])
    while len(out) < n_new:
        if controller is not None:
            h = _entropy(logits_d)
            k = controller.choose_k(h)
        proposals = []
        lg = logits_d
        for _ in range(k):
            tok = int(lg.argmax())
            proposals.append(tok)
            lg = draft(torch.tensor([[tok]], dtype=torch.long), cache=cd)[0, -1]
        stats["draft_forwards"] += k
        # one verification forward: [bonus_prev] + proposals (round 1 merges
        # the prompt prefill)
        inp = (prompt_ids if bonus_prev is None else [bonus_prev]) + proposals
        v = target(torch.tensor([inp], dtype=torch.long), cache=ct)
        stats["target_forwards"] += 1
        offset = len(inp) - k      # V[:, offset - 1 + i] scores proposal i
        amax = v[0].argmax(dim=-1)
        match = 0
        for i in range(k):
            t_i = int(amax[offset - 1 + i])
            if proposals[i] == t_i:
                match += 1
            else:
                break
        bonus = int(amax[offset - 1 + match])
        if match < k:
            ct.truncate(committed + match)
            cd.truncate(committed + match)
        out.extend(proposals[:match])
        out.append(bonus)
        committed += match + 1
        bonus_prev = bonus
        # draft restores its cache to the new committed context
        bt = torch.tensor([[bonus]], dtype=torch.long)
        logits_d = draft(bt, cache=cd)[0, -1]
        stats["draft_forwards"] += 1
        stats["rounds"] += 1
        stats["accepted"] += match
        stats["proposed"] += k
        stats["k_history"].append(k)
        stats["match_history"].append(match)
        if controller is not None:
            stats["entropy_history"].append(h)
            controller.update(match, k, h)
    ids = list(prompt_ids) + out[:n_new]
    return ids, stats
