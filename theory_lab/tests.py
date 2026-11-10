"""Assertion suite. Each fatal defect of the previous version has a test that
would fail on the old code. Pure in-memory computation, no temp files.

    python tests.py -v
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

import baselines as B
import coupled as C
import diffusion as D
import inference as I
import process as P
from metrics import hist, tv

ROOT = Path(__file__).resolve().parent


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.p = P.Params()

    def test_generator_and_invariant_law(self):
        for m in (0.0, 2.0, 7.067, 12.0):
            pi = P.stationary(self.p, m)
            Q = P.generator(self.p, m)
            self.assertLess(np.abs(Q.T @ pi).max(), 1e-12)
            self.assertLess(np.abs(Q.sum(1)).max(), 1e-12)
            self.assertAlmostEqual(pi.sum(), 1.0, places=12)
            self.assertGreater(pi.min(), 0.0)
            gap, zero = P.spectral_gap(self.p, m)
            self.assertLess(abs(zero), 1e-8)
            self.assertGreater(gap, 0.5)

    def test_poisson_derivative_matches_finite_difference(self):
        for m in (0.5, 1.0, 3.0, 7.0, 15.0):
            self.assertLess(abs(P.F_prime_fd(self.p, m) - P.F_prime_poisson(self.p, m)), 1e-6)

    def test_spectral_derivative_bound(self):
        for m in (1.0, 3.0, 7.0, 15.0):
            d = I.derivative_bound(self.p, m)
            self.assertLessEqual(abs(d["Fprime_fd"]), d["derivative_upper_bound"])

    def test_truncation_is_verified_not_asserted(self):
        grid = np.linspace(0, 12, 49)
        tail, _ = P.truncation_report(self.p, grid)
        self.assertLess(tail, self.p.tail_tol)
        q, t2 = P.ensure_truncation(self.p, grid)
        self.assertEqual(q.N, self.p.N)
        self.assertLess(t2, self.p.tail_tol)

    def test_continuous_time_sampling_repairs_jump_chain_bug(self):
        rng = np.random.default_rng(1)
        m = 5.0
        pi = P.stationary(self.p, m)
        s = P.gillespie_time_uniform(self.p, m, 50 + 30000 * 2.0, rng, dt=2.0, burn_in=50.0)
        e = P.embedded_chain(self.p, m, 30000, rng)
        self.assertLess(tv(hist(s, self.p.N), pi), 0.03)
        self.assertGreater(tv(hist(e, self.p.N), pi), 0.08)
        birth, death = P.rates(self.p, m)
        lam = birth + death
        nu = pi * lam
        nu /= nu.sum()
        self.assertLess(tv(hist(e, self.p.N), nu), 0.03)

    def test_ebm_log_partition_sign(self):
        rng = np.random.default_rng(0)
        s = P.gillespie_time_uniform(self.p, 5.0, 50 + 6000 * 2.0, rng, dt=2.0, burn_in=50.0)
        ebm = B.EnergyMLP(self.p.N, seed=0)
        ebm.fit(s, steps=400, lr=5e-3, seed=1)
        self.assertAlmostEqual(float(ebm.log_prob().exp().sum().item()), 1.0, places=5)
        E = torch.full((self.p.N,), -2.0)
        wrong = -(torch.zeros(self.p.N) * (-E - torch.logsumexp(E, 0))).sum()
        self.assertEqual(float(wrong), 0.0)
        loss_at_c = lambda c: float(-(torch.full((self.p.N,), 1.0 / self.p.N)
                                      * (-torch.full((self.p.N,), c)
                                         - torch.logsumexp(torch.full((self.p.N,), c), 0))).sum())
        self.assertLess(loss_at_c(-4.0), loss_at_c(0.0))
        spec = importlib.util.spec_from_file_location("legacy_samplers", ROOT / "_legacy" / "samplers.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        old = mod.EnergySampler(self.p.N, hidden=64, seed=0)
        idx = torch.arange(self.p.N)
        old_norm = float(old.log_prob(idx).exp().sum().item())
        self.assertGreater(abs(old_norm - 1.0), 1e-3)
        E = old.energy(idx).detach()
        wrong_form = -E - torch.logsumexp(E, dim=0)
        self.assertLess(float((old.log_prob(idx).detach() - wrong_form).abs().max()), 1e-5)


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.p = P.Params()

    def test_matrix_exponential(self):
        A = np.array([[0.0, 2.0], [-1.0, -3.0]])
        E = D.expm(A)
        self.assertLess(np.abs(D.expm(A / 2) @ D.expm(A / 2) - E).max(), 1e-9)

    def test_semigroup_and_row_stochastic(self):
        kp = D.physical_kernel(self.p, 5.5, T=12)
        ku = D.uniform_kernel(self.p.N, T=12)
        for k in (kp, ku):
            self.assertLess(np.abs(k.Pbar.sum(2) - 1.0).max(), 1e-9)
            self.assertLess(np.abs(k.Pstep.sum(2) - 1.0).max(), 1e-9)
            for j in range(1, k.T + 1):
                self.assertLess(np.abs(k.Pbar[j - 1] @ k.Pstep[j] - k.Pbar[j]).max(), 1e-9)
            self.assertLess(0.5 * np.abs(k.Pbar[k.T] - 1.0 / k.N).sum(1).max(), 0.01)

    def test_exact_bayes_reverse_kernel(self):
        k = D.physical_kernel(self.p, 5.5, T=10)
        net = D.CategoricalDiffusion(self.p.N, hidden=32, seed=0)
        rng = np.random.default_rng(0)
        x = rng.integers(0, self.p.N, 64)
        pr = net.transition_probs(x, 7, k, 0.0)
        self.assertLess(np.abs(pr.sum(1) - 1.0).max(), 1e-9)
        self.assertGreaterEqual(pr.min(), 0.0)

    def test_spectral_mixing_bound_holds(self):
        for k in (D.physical_kernel(self.p, 5.5, T=40),
                  D.uniform_kernel(self.p.N, T=40)):
            mx, _, bd = D.mixing_curve(k)
            self.assertTrue(bool(np.all(mx <= bd + 1e-9)))
            self.assertLess(mx[-1], 0.01)

    def test_weyl_link_between_both_spectral_gaps(self):
        for m in (1.0, 5.5, 9.5, 12.0):
            r = D.weyl_gap_relation(self.p, m)
            self.assertTrue(r["weyl_bound_holds"])
            self.assertGreater(r["gap_Q"], 0.5)
            self.assertGreater(r["gap_R"], 0.0)
            q = D.physical_gap_bounds(self.p, m)
            self.assertTrue(q["bound_holds"])
            self.assertGreater(q["lower_bound"], 0.0)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.p = P.Params()

    def test_bistable_window_and_stability(self):
        fps = P.fixed_points(self.p)
        self.assertEqual(len(fps), 3)
        self.assertEqual(sum(r["stable"] for r in fps), 2)
        pbig = self.p.update(N=128)
        window, _ = P.bifurcation_window(pbig, np.arange(8.0, 14.01, 0.25))
        self.assertIsNotNone(window)
        self.assertLess(window["beta_left"], 10.0)
        self.assertGreater(window["beta_right"], 10.0)
        self.assertLess(window["width"], 3.0)

    def test_critical_slowing_down(self):
        q = self.p.update(beta=9.124)
        fps = P.fixed_points(q, hi=40)
        L = max(abs(r["Fprime"]) for r in fps)
        self.assertGreater(L, 0.95)
        b1 = I.propagation_bound(self.p, 1.137, 0.01)["bound"]
        b2 = I.propagation_bound(q, min(fps, key=lambda r: abs(abs(r["Fprime"]) - 1))["m"], 0.01)["bound"]
        self.assertGreater(b2, b1)
        self.assertTrue(np.isfinite(b1))

    def test_importance_sampling_unbiased(self):
        pi = P.stationary(self.p, 7.067)
        k = 30
        q = I.tilted_proposal(self.p.N, k, 2.0)
        e, se, _ = I.is_estimate(pi, q, 200000, np.random.default_rng(3))
        self.assertLess(abs(e - float(pi[k:].sum())), max(5 * se, 1e-7))
        res = I.repeated_is(pi, q, 2000, 100, np.random.default_rng(4))
        self.assertLess(abs(res["bias"]), 4 * res["mean_se"] / np.sqrt(100))
        self.assertGreater(res["coverage"], 0.85)
        self.assertLess(res["mse"], I.crude_mse(pi, k, 2000))

    def test_zero_variance_proposal(self):
        pi = P.stationary(self.p, 7.067)
        k = 30
        q = np.zeros(self.p.N)
        q[k:] = pi[k:] / pi[k:].sum()
        n, cv2 = I.required_n(pi, q, 0.05)
        self.assertLess(cv2, 1e-10)
        self.assertAlmostEqual(n, 400.0, places=4)

    def test_coupled_invariant_law(self):
        cp = C.CoupledParams(Nc=16)
        Q = C.generator_coupled(cp)
        pi, res = C.stationary_numeric(Q)
        self.assertLess(res, 1e-10)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=10)
        self.assertGreaterEqual(pi.min(), 0.0)


class ProjectHygieneTests(unittest.TestCase):
    def test_no_hardcoded_output_path(self):
        src = (ROOT / "experiments.py").read_text(encoding="utf-8")
        self.assertNotIn("2026-09-10", src)
        self.assertIn("Path(__file__).resolve().parent", src)

    def test_results_are_relative_to_project(self):
        from experiments import OUT
        self.assertEqual(OUT.parent.resolve(), ROOT.resolve())


if __name__ == "__main__":
    unittest.main(verbosity=2)
