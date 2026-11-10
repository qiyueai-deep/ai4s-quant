"""生成式采样器：能量模型（EBM，Lyapunov 型势函数）与离散扩散（D3PM）。

设定与真实场景一致：采样器只能从 Monte Carlo 样本学习，不使用精确平稳分布。
EBM 的配分函数在截断状态空间上可精确求和，因此是严格的极大似然训练。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(d_in, hidden, d_out, layers=3):
    mods = [nn.Linear(d_in, hidden), nn.SiLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.SiLU()]
    mods += [nn.Linear(hidden, d_out)]
    return nn.Sequential(*mods)


class EnergySampler:
    """p(i) ∝ exp(-E_theta(i))。能量由状态 embedding + MLP 参数化。"""

    def __init__(self, N, hidden=64, seed=0):
        torch.manual_seed(seed)
        self.N = N
        self.emb = nn.Embedding(N, 16)
        self.net = make_mlp(16, hidden, 1, layers=3)
        for m in list(self.emb.parameters()) + list(self.net.parameters()):
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def energy(self, x_idx):
        return self.net(self.emb(x_idx)).squeeze(-1)

    def log_prob(self, x_idx):
        e_all = self.energy(torch.arange(self.N))
        logZ = torch.logsumexp(e_all, dim=0)
        e_x = self.energy(x_idx)
        return -e_x - logZ

    def fit(self, samples, steps=6000, lr=5e-4, batch=1024, wd=0.0, seed=1):
        torch.manual_seed(seed)
        samples = torch.as_tensor(np.asarray(samples), dtype=torch.long)
        h = torch.bincount(samples, minlength=self.N).float()
        q = h / h.sum()
        params = list(self.emb.parameters()) + list(self.net.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        last = float("nan")
        for step in range(steps):
            e_all = self.energy(torch.arange(self.N))
            logp = -e_all - torch.logsumexp(e_all, dim=0)
            loss = -(q * logp).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            last = float(loss.item())
        return last

    @torch.no_grad()
    def sample(self, n, rng):
        probs = F.softmax(-self.energy(torch.arange(self.N)), dim=0).numpy()
        return rng.choice(self.N, size=n, p=probs)


class DiscreteDiffusion:
    """D3PM：均匀腐蚀核 + 学 x0 预测，支持标量条件 cond（平均场实验用）。"""

    def __init__(self, N, hidden=128, T=60, seed=0):
        torch.manual_seed(seed)
        self.N = N
        self.T = T
        self.betas = torch.linspace(1e-4, 0.06, T + 1)
        self.net = make_mlp(N + 2, hidden, N, layers=3)

    def _inp(self, x, t, cond):
        """t、cond 可为标量或与 x 等长的向量。"""
        x = torch.as_tensor(x, dtype=torch.long)
        t = torch.as_tensor(t, dtype=torch.float32)
        cond = torch.as_tensor(cond, dtype=torch.float32)
        if t.dim() == 0:
            t = t.expand(len(x))
        if cond.dim() == 0:
            cond = cond.expand(len(x))
        xoh = F.one_hot(x, self.N).float()
        tvec = (t / self.T).unsqueeze(1)
        cv = cond.unsqueeze(1)
        return torch.cat([xoh, tvec, cv], dim=1)

    def fit(self, samples, conds=None, steps=3000, lr=1e-3, batch=1024, seed=1):
        torch.manual_seed(seed)
        samples = torch.as_tensor(np.asarray(samples), dtype=torch.long)
        if conds is None:
            conds = np.zeros(len(samples))
        conds = torch.as_tensor(np.asarray(conds), dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        n = len(samples)
        last = float("nan")
        for step in range(steps):
            idx = torch.randint(0, n, (min(batch, n),))
            x0, c = samples[idx], conds[idx]
            t = torch.randint(1, self.T + 1, (len(x0),))
            beta = self.betas[t]
            corrupt = torch.rand(len(x0)) < beta
            noise = torch.randint(0, self.N, (len(x0),))
            xt = torch.where(corrupt, noise, x0)
            logits = self.net(self._inp(xt, t.float(), c))
            loss = F.cross_entropy(logits, x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        return last

    @torch.no_grad()
    def sample(self, n, rng, cond=0.0):
        x = torch.as_tensor(rng.integers(0, self.N, size=n).astype(np.int64))
        for t in range(self.T, 0, -1):
            logits = self.net(self._inp(x, t, cond))
            x0_hat = torch.distributions.Categorical(logits=logits).sample()
            beta = float(self.betas[t])
            keep = torch.rand(len(x)) >= beta
            noise = torch.randint(0, self.N, (len(x),))
            x = torch.where(keep, x0_hat, noise)
        return x.numpy().astype(np.int64)
