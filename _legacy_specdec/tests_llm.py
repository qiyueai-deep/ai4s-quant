"""Assertion suite for the self-contained speculative-decoding project.

    python tests_llm.py -v
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from specdec import theory as T
from specdec.core import TinyTransformerLM, load_artifacts
from specdec.data import generate_corpus
from specdec.decode import AdaptiveController, greedy_generate, speculative_generate
from specdec.probes import probe_distributions

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


class TheoryTests(unittest.TestCase):
    def test_acceptance_identity_is_exact(self):
        rng = np.random.default_rng(0)
        for _ in range(60):
            p, q = rng.dirichlet(np.ones(9)), rng.dirichlet(np.ones(9))
            self.assertAlmostEqual(T.acceptance_from_tv(p, q),
                                   float(np.minimum(p, q).sum()), places=12)
            self.assertAlmostEqual(T.acceptance_from_tv(p, q),
                                   1.0 - T.tv_distance(p, q), places=12)

    def test_acceptance_monte_carlo_agrees(self):
        rng = np.random.default_rng(1)
        p, q = rng.dirichlet(np.ones(12) * 0.7), rng.dirichlet(np.ones(12) * 0.7)
        mc = T.acceptance_monte_carlo(p, q, rng, n=400000)
        self.assertLess(abs(mc - T.acceptance_from_tv(p, q)), 0.005)

    def test_pinsker_bound(self):
        rng = np.random.default_rng(2)
        for _ in range(60):
            p, q = rng.dirichlet(np.ones(9) * 0.5), rng.dirichlet(np.ones(9) * 0.5)
            self.assertLessEqual(T.tv_distance(p, q),
                                 np.sqrt(T.kl_divergence(p, q) / 2.0) + 1e-9)
            self.assertLessEqual(T.pinsker_alpha_lower_bound(T.kl_divergence(p, q)),
                                 T.acceptance_from_tv(p, q) + 1e-9)

    def test_expected_tokens_boundaries(self):
        self.assertAlmostEqual(T.expected_tokens(0.0, 10), 1.0, places=12)
        self.assertAlmostEqual(T.expected_tokens(1.0 - 1e-13, 10), 11.0, places=6)
        vals = [T.expected_tokens(0.9, k) for k in range(1, 12)]
        self.assertTrue(all(a <= b for a, b in zip(vals, vals[1:])))
        self.assertLess(vals[-1], 1.0 / 0.1)

    def test_finite_optimum_is_provable(self):
        self.assertTrue(T.has_finite_optimum(0.9, 0.1, 2.0))
        self.assertFalse(T.has_finite_optimum(1.0, 0.1, 2.0))
        k, _, scores = T.optimal_k(0.9, 0.1, Kmax=64, c0=2.0)
        self.assertTrue(1 < k < 64)
        self.assertLess(scores[-1], float(scores.max()))
        far = T.speedup_model(0.9, 5000, 0.1, 2.0)
        self.assertLess(far, scores[0])

    def test_cost_model_calibration_recovers_parameters(self):
        alpha, c0, g = 0.9, 2.0, 0.1
        Ks = np.arange(1, 33, 4)
        S = np.array([T.speedup_model(alpha, int(k), g, c0) for k in Ks])
        fit = T.fit_cost_model(alpha, Ks, S)
        self.assertAlmostEqual(fit["c0"], c0, places=8)
        self.assertAlmostEqual(fit["g"], g, places=8)
        self.assertGreater(fit["r2"], 0.999)

    def test_distillation_gain(self):
        out = T.distillation_gain(0.387, 0.109)
        self.assertAlmostEqual(out["before"], 1 - np.sqrt(0.387 / 2), places=12)
        self.assertAlmostEqual(out["after"], 1 - np.sqrt(0.109 / 2), places=12)
        self.assertGreater(out["after"], out["before"])
        self.assertGreater(out["delta"], 0.15)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.target, cls.draft, cls.tokenizer = load_artifacts(ART)
        cls.manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))

    def test_checkpoints_match_manifest(self):
        self.assertEqual(self.target.parameter_count,
                         self.manifest["target_parameters"])
        self.assertEqual(self.draft.parameter_count,
                         self.manifest["draft_parameters"])
        self.assertEqual(self.tokenizer.vocab_size, self.manifest["vocab_size"])
        self.assertGreater(self.target.parameter_count /
                           self.draft.parameter_count, 5.0)

    def test_target_is_trained(self):
        ce = self.manifest["metrics"]["val_ce_target"]
        vocab = self.tokenizer.vocab_size
        self.assertLess(ce, 0.35 * np.log(vocab))

    def test_distillation_beats_sft_on_kl(self):
        m = self.manifest["metrics"]
        self.assertLess(m["kl_target_to_draft_distilled"],
                        m["kl_target_to_draft_sft"])

    def test_real_probes_satisfy_the_identity(self):
        rows = probe_distributions(self.target, self.draft, self.tokenizer,
                                   n_prompts=2, positions=4, seed=0)
        self.assertEqual(len(rows), 8)
        for r in rows:
            self.assertGreaterEqual(r["alpha"], 0.0)
            self.assertLessEqual(r["alpha"], 1.0)
            self.assertAlmostEqual(r["alpha"], r["alpha_exact"], places=10)
            self.assertGreaterEqual(r["alpha"] + 1e-9,
                                    T.pinsker_alpha_lower_bound(r["kl_pq"]))


class DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.target, cls.draft, cls.tokenizer = load_artifacts(ART)
        text = (ROOT / "data" / "corpus_val.txt").read_text(encoding="utf-8")
        cls.prompts = [cls.tokenizer.encode(ln[:16], add_bos=True)
                       for ln in text.splitlines()[:3]]

    def test_kv_cache_matches_full_forward(self):
        model = self.target
        ids = self.prompts[0]
        with torch.inference_mode():
            full = model(torch.tensor([ids]), cache=None)
            cache = model.new_cache()
            inc = [model(torch.tensor([[t]]), cache=cache) for t in ids]
        for i in range(len(ids)):
            diff = float((full[0, i] - inc[i][0, 0]).abs().max())
            self.assertLess(diff, 1e-4)

    def test_speculative_output_equals_greedy_exactly(self):
        ref = [greedy_generate(self.target, p, 48) for p in self.prompts]
        for k in (1, 2, 4, 8, 16, 32):
            for p, r in zip(self.prompts, ref):
                out, _ = speculative_generate(self.target, self.draft, p, 48, k=k)
                self.assertEqual(out, r, msg=f"K={k}")
        for bucketed in (False, True):
            ctrl = AdaptiveController(self.target, self.draft,
                                      entropy_bucketing=bucketed, timing_repeats=10)
            for p, r in zip(self.prompts, ref):
                out, _ = speculative_generate(self.target, self.draft, p, 48,
                                              controller=ctrl)
                self.assertEqual(out, r)

    def test_benchmark_outputs_all_match_greedy(self):
        bench = json.loads((ART / "benchmark.json").read_text(encoding="utf-8"))
        self.assertTrue(all(r["outputs_match_greedy"] for r in bench["speculative"]))
        self.assertTrue(all(r["outputs_match_greedy"] for r in bench["adaptive"]))


class ControllerTests(unittest.TestCase):
    def test_beta_estimator_tracks_synthetic_rates(self):
        torch.set_num_threads(1)
        target, draft, _ = load_artifacts(ART)
        ctrl = AdaptiveController(target, draft, timing_repeats=5)
        rng = np.random.default_rng(0)
        for _ in range(200):
            ctrl.update(int(rng.random() < 0.9), 1, 0.0)
        self.assertGreater(ctrl.acc["low"], 0.75)
        self.assertLess(ctrl.acc["low"], 0.98)
        for _ in range(200):
            ctrl.update(int(rng.random() < 0.4), 1, 0.0)
        self.assertLess(ctrl.acc["low"], 0.6)

    def test_bucketed_estimator_separates_contexts(self):
        torch.set_num_threads(1)
        target, draft, _ = load_artifacts(ART)
        ctrl = AdaptiveController(target, draft, entropy_bucketing=True,
                                  timing_repeats=5)
        rng = np.random.default_rng(1)
        for _ in range(100):
            ctrl.update(int(rng.random() < 0.9), 1, 0.2)
            ctrl.update(int(rng.random() < 0.3), 1, 2.5)
        self.assertGreater(ctrl.acc["low"], ctrl.acc["high"] + 0.2)
        self.assertGreater(ctrl.acc["low"], 0.8)
        self.assertLess(ctrl.acc["high"], 0.5)

    def test_choose_k_stays_on_grid(self):
        torch.set_num_threads(1)
        target, draft, _ = load_artifacts(ART)
        ctrl = AdaptiveController(target, draft, timing_repeats=5)
        for h in (0.0, 0.5, 1.0, 3.0):
            self.assertIn(ctrl.choose_k(h), ctrl.k_grid)


class CorpusTests(unittest.TestCase):
    def test_corpus_is_deterministic(self):
        a = generate_corpus(seed=0, n_lines=200)
        b = generate_corpus(seed=0, n_lines=200)
        c = generate_corpus(seed=1, n_lines=200)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_training_smoke_reduces_loss(self):
        torch.manual_seed(0)
        from specdec.core import ModelConfig
        from specdec.train import build_tokenizer, encode_file, init_weights, sample_batch
        text = generate_corpus(seed=5, n_lines=120)
        tok = build_tokenizer(text)
        cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=64,
                          d_model=32, n_layers=1, n_heads=2, d_ff=64)
        model = TinyTransformerLM(cfg)
        init_weights(model)
        ids = encode_file(tok, text)
        rng = np.random.default_rng(0)
        x = sample_batch(ids, 8, 24, rng)
        from torch import nn
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = None
        for step in range(40):
            logits = model(x[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if first is None:
                first = float(loss.detach())
        self.assertLess(float(loss.detach()), first - 1.0)


class HygieneTests(unittest.TestCase):
    def test_no_external_references(self):
        for name in ("experiments_llm.py", "specdec/train.py",
                     "specdec/decode.py", "specdec/benchmark.py",
                     "specdec/data.py"):
            src = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("SpecForge", src)
            self.assertNotIn("2026-09-10", src)
            self.assertNotIn("import transformers", src)

    def test_artifacts_are_self_owned(self):
        manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("corpus_train_chars", manifest)
        self.assertNotIn("benchmark_new_tokens", manifest)
        for f in ("target.pt", "draft.pt", "tokenizer.json", "manifest.json",
                  "benchmark.json"):
            self.assertTrue((ART / f).exists(), f)
        for f in ("specforge_manifest.json", "specforge_benchmark.json"):
            self.assertFalse((ART / f).exists(), f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
