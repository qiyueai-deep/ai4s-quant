"""Assertion suite for the quantization project.

    python tests_quant.py -v
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from lmlab.data import generate_corpus_hard
from lmlab.model import load_checkpoint, CharTokenizer
from quant import theory as T
from quant.calibration import quantizable_linears
from quant.gptq import gptq_quantize
from quant.pipeline import eval_ce, quantize_model
from quant.rtn import quantize_rows, quantize_rows_clip

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


class TheoryTests(unittest.TestCase):
    def test_uniform_rounding_variance(self):
        rng = np.random.default_rng(0)
        delta = 0.37
        x = (rng.uniform(-5, 5, 2_000_000) / delta)
        err = (torch.round(torch.tensor(x)) - torch.tensor(x)).numpy() * delta
        self.assertLess(abs(float(np.var(err)) - delta ** 2 / 12.0)
                        / (delta ** 2 / 12.0), 0.01)

    def test_predicted_recon_mse_matches_simulation(self):
        rng = np.random.default_rng(1)
        out_f, in_f = 16, 24
        W = torch.tensor(rng.normal(0, 0.3, (out_f, in_f)))
        X = torch.tensor(rng.normal(0, 1.0, (4000, in_f)))
        H = 2.0 * X.T @ X
        pred = T.predicted_recon_mse(W, 3, H, "per_channel")
        Wq, _ = quantize_rows(W, 3)
        meas = T.measured_recon_mse(W, Wq, H)
        self.assertLess(abs(np.log(meas / pred["predicted_mse"])), 0.35)

    def test_layer_cost_scales_as_four_per_bit(self):
        c = T.layer_cost(3.0, 100.0, 5)
        self.assertAlmostEqual(c / T.layer_cost(3.0, 100.0, 4), 0.25, places=12)

    def test_allocate_bits_beats_greedy_by_brute_force(self):
        rng = np.random.default_rng(2)
        stats = [dict(n_params=int(rng.integers(50, 200)),
                      scales_sum2=float(rng.uniform(0.01, 0.2)),
                      frob=float(rng.uniform(500, 5000))) for _ in range(3)]
        budget = sum(s["n_params"] * 2 for s in stats) + sum(
            s["n_params"] for s in stats)  # room for ~1 bit above minimum
        bits, memory = T.allocate_bits(stats, budget)
        self.assertLessEqual(memory, budget)
        # brute force over all allocations within budget
        n = len(stats)
        best = None
        import itertools
        ranges = [range(2, 7) for _ in range(n)]
        for combo in itertools.product(*ranges):
            mem = sum(stats[i]["n_params"] * combo[i] for i in range(n))
            if mem > budget:
                continue
            cost = sum(T.layer_cost(stats[i]["scales_sum2"],
                                    stats[i]["frob"], combo[i]) for i in range(n))
            if best is None or cost < best:
                best = cost
        greedy_cost = sum(T.layer_cost(stats[i]["scales_sum2"],
                                       stats[i]["frob"], bits[i]) for i in range(n))
        self.assertLessEqual(greedy_cost, best * (1 + 1e-9))

    def test_kkt_marginal_gap_bounded(self):
        rng = np.random.default_rng(3)
        stats = [dict(n_params=int(rng.integers(50, 200)),
                      scales_sum2=float(rng.uniform(0.01, 0.2)),
                      frob=float(rng.uniform(500, 5000))) for _ in range(4)]
        budget = sum(s["n_params"] * 3 for s in stats)
        bits, _ = T.allocate_bits(stats, budget)
        self.assertLessEqual(T.kkt_marginal_gap(stats, bits), 4.0 + 1e-9)


class GPTQTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.out_f, self.in_f = 32, 48
        self.W = torch.tensor(rng.normal(0, 0.2, (self.out_f, self.in_f)))
        X = torch.tensor(rng.normal(0, 1.0, (3000, self.in_f)))
        self.H = 2.0 * X.T @ X

    def test_surrogate_identity(self):
        Hd = self.H.double() + 0.01 * torch.diag(self.H.double()).mean() \
            * torch.eye(self.in_f, dtype=torch.float64)
        Wq, info = gptq_quantize(self.W, self.H, 3)
        meas = float(torch.trace((self.W.double() - Wq.double()).T
                                 @ (self.W.double() - Wq.double()) @ Hd))
        self.assertAlmostEqual(info["surrogate"] / meas, 1.0, places=6)

    def test_gptq_never_worse_than_rtn(self):
        Wq_r, _ = quantize_rows(self.W, 3)
        Wq_g, _ = gptq_quantize(self.W, self.H, 3)
        self.assertLessEqual(T.measured_recon_mse(self.W, Wq_g, self.H),
                             T.measured_recon_mse(self.W, Wq_r, self.H) + 1e-9)

    def test_identity_hessian_reduces_to_rtn(self):
        H = torch.eye(self.in_f, dtype=torch.float64)
        Wq_g, _ = gptq_quantize(self.W, H, 3)
        Wq_r, _ = quantize_rows(self.W, 3)
        self.assertTrue(torch.allclose(Wq_g.double(), Wq_r.double(), atol=1e-6))

    def test_clip_search_never_worse_than_plain(self):
        Wq_r, _ = quantize_rows(self.W, 2)
        Wq_c, s, pct = quantize_rows_clip(self.W, 2, self.H)
        self.assertLessEqual(T.measured_recon_mse(self.W, Wq_c, self.H),
                             T.measured_recon_mse(self.W, Wq_r, self.H) + 1e-9)
        self.assertTrue(bool((pct <= 100.0).all()))
        self.assertTrue(bool((s > 0).all()))

    def test_clip_prediction_with_mixture(self):
        Wq_c, s, pct = quantize_rows_clip(self.W, 2, self.H)
        pred = T.predicted_recon_mse(self.W, 2, self.H, scales=s)
        meas = T.measured_recon_mse(self.W, Wq_c, self.H)
        self.assertLess(abs(np.log(meas / pred["predicted_mse"])), 0.6)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = load_checkpoint(ART / "target_q.pt")
        cls.tokenizer = CharTokenizer.load(ART / "tokenizer_q.json")
        cls.manifest = json.loads(
            (ART / "manifest_q.json").read_text(encoding="utf-8"))
        from quant.calibration import collect_calibration
        cls.calib = collect_calibration(cls.model, cls.tokenizer, seed=0)

    def test_checkpoint_matches_manifest(self):
        self.assertEqual(self.model.parameter_count,
                         self.manifest["parameters"])
        self.assertEqual(self.tokenizer.vocab_size,
                         self.manifest["vocab_size"])
        self.assertLess(self.manifest["val_ce"], 1.5)

    def test_eval_deterministic(self):
        a = eval_ce(self.model, self.tokenizer, seed=0)
        b = eval_ce(self.model, self.tokenizer, seed=0)
        self.assertEqual(a["ce"], b["ce"])

    def test_quantize_model_changes_weights_but_keeps_shape(self):
        qm, info = quantize_model(self.model, 3, "gptq", "per_channel",
                                  calibration=self.calib)
        self.assertEqual(len(info), len(list(quantizable_linears(qm))))
        for (n1, m1), (n2, m2) in zip(quantizable_linears(self.model),
                                      quantizable_linears(qm)):
            self.assertEqual(n1, n2)
            self.assertFalse(torch.equal(m1.weight.data, m2.weight.data))
            self.assertEqual(m1.weight.shape, m2.weight.shape)

    def test_bigger_bits_closer_to_base(self):
        base = eval_ce(self.model, self.tokenizer, seed=0)
        q8, _ = quantize_model(self.model, 8, "rtn", "per_channel")
        q2, _ = quantize_model(self.model, 2, "rtn", "per_channel")
        e8 = eval_ce(q8, self.tokenizer, seed=0)
        e2 = eval_ce(q2, self.tokenizer, seed=0)
        self.assertLess(abs(e8["ce"] - base["ce"]), abs(e2["ce"] - base["ce"]))


class HygieneTests(unittest.TestCase):
    def test_artifacts_present(self):
        for f in ("target_q.pt", "tokenizer_q.json", "manifest_q.json"):
            self.assertTrue((ART / f).exists(), f)

    def test_no_external_references(self):
        for name in ("experiments_quant.py", "lmlab/train.py",
                     "lmlab/data.py", "lmlab/model.py",
                     "quant/gptq.py", "quant/rtn.py", "quant/theory.py"):
            src = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("SpecForge", src)
            self.assertNotIn("2026-09-10", src)
            self.assertNotIn("import transformers", src)

    def test_gptq_attribution_present(self):
        src = (ROOT / "quant" / "gptq.py").read_text(encoding="utf-8")
        self.assertIn("Frantar", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
