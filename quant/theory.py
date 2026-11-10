"""Closed-form theory predictions for weight quantization, all checkable.

T1. Uniform rounding variance. For weights that cover the quantizer levels
    uniformly the rounding error is uniform on [-Delta/2, Delta/2] with
    variance Delta^2 / 12. This is the high-bits limit; at low bits it fails
    badly (see T2).

T2. Layer reconstruction MSE. For a linear layer with weights W (out, in),
    calibration inputs X (n, in) and perturbation dW,
        E ||dW x||^2 = trace(dW^T dW * X^T X).
    Modelling each row's weights as Gaussian with their own mean and
    standard deviation, the per-row rounding-error variance E[(w - Q(w))^2]
    is evaluated exactly by 1-D quadrature of the true error function
    g(w) = w - clip(round(w/Delta), -L, L) * Delta (clipping included), so
        MSE_pred = (sum_r sigma_r^2) * ||X||_F^2.
    Everything on the right is computed from the weights and the calibration
    Hessian only -- no forward pass needed after quantization.

T3. GPTQ surrogate. Each GPTQ column step decreases the total second-order
    surrogate exactly by sum_rows (w - q_i)^2 / U[i,i]^2, so the final
    surrogate is the sum of the per-column terms; tests verify the identity
    against a direct computation, experiments verify GPTQ <= RTN.

T4. Water-filling bit allocation. The per-layer cost at b bits is
        cost_l(b) = (sum_r sigma_{l,r}(b)^2 / 12) * ||X_l||_F^2,
    and sigma_l(b) scales as 2^{-(b - b_init)}, so cost decays 4x per bit.
    Minimizing total cost under a total-memory budget is an integer program;
    greedy marginal allocation (give the next bit-unit to the layer with the
    largest marginal decrease per memory unit) is optimal here because each
    layer's cost is convex decreasing in b, and the KKT condition of the
    relaxation is that remaining marginals are equalised. The tests verify
    the greedy solution up to one single-bit swap against brute force.
"""
from __future__ import annotations

import torch

from .rtn import levels, quantize_rows, quantize_tensor

_Z = torch.linspace(-8.0, 8.0, 4001, dtype=torch.float64)
_PHI = torch.exp(-0.5 * _Z ** 2)
_PHI = _PHI / _PHI.sum()


def rounding_var(bits, scale):
    """Delta^2 / 12 for step Delta = scale."""
    return float(scale) ** 2 / 12.0


def row_error_moments(W, bits, granularity="per_channel", scales=None):
    """Per-row first two moments of the rounding error under a Gaussian model
    of each row's weights: returns (var_r, bias_r, clip_frac) where
    var_r = Var[g(w)], bias_r = E[g(w)] and g is the true error function
    g(w) = w - clip(round(w/Delta), -L, L) * Delta, evaluated by 1-D
    quadrature.

    The naive closed form Delta^2 / 12 assumes the weights cover the levels
    uniformly AND that the error is zero-mean; at low bits with max-abs
    scaling most weights are far smaller than Delta and round straight to
    zero (error = w itself), and the nonzero bias couples coordinates through
    the off-diagonal of the Hessian. Both effects are exactly captured here."""
    Wd = W.double()
    mu = Wd.mean(dim=1)
    tau = Wd.std(dim=1, unbiased=False)
    if scales is None:
        if granularity == "per_channel":
            _, scales = quantize_rows(W, bits)
        elif granularity == "per_tensor":
            _, scales = quantize_tensor(W, bits)
            scales = scales.expand(W.shape[0])
        else:
            raise ValueError(granularity)
    scales = scales.double()
    L = float(levels(bits))
    w = mu.unsqueeze(1) + tau.unsqueeze(1) * _Z.unsqueeze(0)      # (out, nodes)
    q = torch.clamp(torch.round(w / scales.unsqueeze(1)), -L, L) \
        * scales.unsqueeze(1)
    g = w - q
    mean_g = (g * _PHI.unsqueeze(0)).sum(dim=1)                    # (out,)
    second_g = ((g ** 2) * _PHI.unsqueeze(0)).sum(dim=1)
    var = second_g - mean_g ** 2
    clip_frac = float((((w.abs() > L * scales.unsqueeze(1)).double())
                       * _PHI.unsqueeze(0)).sum() / W.shape[0])
    return var, mean_g, clip_frac


def predicted_recon_mse(W, bits, H, granularity="per_channel", scales=None):
    """T2: prediction of trace(dW^T dW * H) for RTN with or without
    clip-search scales. Decomposes the error into per-coordinate variances
    (diagonal of the Hessian) and squared biases (full Hessian):
        E[d^T H d] = sum_j v_j H_jj + b^T H b.
    Everything is computed from the weights and the calibration Hessian
    only -- no forward pass needed after quantization."""
    var, bias, clip_frac = row_error_moments(W, bits, granularity,
                                             scales=scales)
    Hd = H.double()
    # per row r, E[d_r d_r^T] = v_r I + b_r^2 * 1 1^T (same marginal for every
    # input coordinate of the row), so
    #   E[d_r^T H d_r] = v_r trace(H) + b_r^2 * (1^T H 1)
    pred = float(var.sum() * torch.trace(Hd)
                 + bias.sum() ** 2 * Hd.sum())
    return dict(predicted_mse=pred, clip_fraction=clip_frac)


def alignment_factor(W, H):
    """T2b. How much a weight row's direction aligns with the input
    covariance's top eigenspace:
        kappa_r = (w_r^T H w_r) / (||w_r||^2 * trace(H) / in_features).
    kappa = 1 for isotropic weight directions; trained weight rows align with
    the high-variance input directions, giving kappa > 1. This is exactly the
    factor by which the isotropic prediction under-estimates a 2-bit error
    whose direction is the weight direction itself."""
    Hd = H.double()
    Wd = W.double()
    quad = ((Wd @ Hd) * Wd).sum(dim=1)                 # w_r^T H w_r
    norms = (Wd ** 2).sum(dim=1)
    return float((quad / norms.clamp_min(1e-300)).mean()
                 / (torch.trace(Hd) / W.shape[1]))


def measured_recon_mse(W, Wq, H):
    """Exact measured surrogate trace(dW^T dW * H) with dW = W - Wq."""
    D = W.double() - Wq.double()
    return float(torch.trace(D.T @ D @ H.double()))


def layer_cost(scales_sum2, frob, bits):
    """cost_l(b) = sum_r sigma_{l,r}(b)^2 / 12 * ||X_l||_F^2, with sigma at
    bit b equal to sigma at the reference bit 8 times 2^{-(b-8)}; scales_sum2
    is sum_r sigma_{l,r}(8)^2."""
    return scales_sum2 * (0.25 ** (bits - 8)) / 12.0 * frob


def allocate_bits(layer_stats, total_memory, cost_fn=None, b_min=2, b_max=8):
    """Bit allocation under a total-memory budget.

    layer_stats: list of dicts with key 'n_params' (plus 'scales_sum2' and
    'frob' when the default cost model is used). cost_fn(i, b) returns the
    layer's reconstruction cost at b bits; the default is the closed-form
    4x-per-bit model (T4). A measured table (e.g. GPTQ surrogates computed on
    calibration) can be passed instead -- that is what the experiments use.

    Phase 1 is greedy marginal allocation (decreasing marginals make greedy
    near-optimal). Phase 2 is an exchange local search: repeatedly move one
    bit-unit from a layer to another whenever it lowers the total cost within
    the budget. Pure greedy can strand memory fragments (it stops when no
    single layer's next bit fits), which measurably loses to the exchange.
    """
    n = len(layer_stats)
    if cost_fn is None:
        cost_fn = lambda i, b: layer_cost(layer_stats[i]["scales_sum2"],
                                          layer_stats[i]["frob"], b)
    bits = {i: b_min for i in range(n)}
    memory = sum(layer_stats[i]["n_params"] * b_min for i in range(n))
    if memory > total_memory:
        raise ValueError("budget below the minimum-bit configuration")

    while True:
        best_i, best_gain = None, 0.0
        for i in range(n):
            if bits[i] >= b_max:
                continue
            if memory + layer_stats[i]["n_params"] > total_memory:
                continue
            gain = (cost_fn(i, bits[i]) - cost_fn(i, bits[i] + 1)) \
                / layer_stats[i]["n_params"]
            if gain > best_gain:
                best_i, best_gain = i, gain
        if best_i is None:
            break
        bits[best_i] += 1
        memory += layer_stats[best_i]["n_params"]

    improved = True
    while improved:
        improved = False
        for i in range(n):
            if bits[i] <= b_min:
                continue
            for j in range(n):
                if j == i or bits[j] >= b_max:
                    continue
                if memory - layer_stats[i]["n_params"] \
                        + layer_stats[j]["n_params"] > total_memory:
                    continue
                delta = (cost_fn(i, bits[i] - 1) - cost_fn(i, bits[i])) \
                    + (cost_fn(j, bits[j] + 1) - cost_fn(j, bits[j]))
                if delta < -1e-12:
                    bits[i] -= 1
                    bits[j] += 1
                    memory += layer_stats[j]["n_params"] \
                        - layer_stats[i]["n_params"]
                    improved = True
    memory = sum(layer_stats[i]["n_params"] * bits[i] for i in range(n))
    return bits, memory


def kkt_marginal_gap(layer_stats, bits, cost_fn=None):
    """Ratio between the largest and smallest remaining marginal gains.
    1.0 at the continuous optimum; for the default 4x-per-bit cost model the
    integer greedy stops within a factor of 4 of equalised marginals."""
    if cost_fn is None:
        cost_fn = lambda i, b: layer_cost(layer_stats[i]["scales_sum2"],
                                          layer_stats[i]["frob"], b)

    def marginal(i):
        b = bits[i]
        return (cost_fn(i, b) - cost_fn(i, b + 1)) / layer_stats[i]["n_params"]

    marg = [marginal(i) for i in range(len(layer_stats))]
    hi, lo = max(marg), min(marg)
    return hi / max(lo, 1e-300)
