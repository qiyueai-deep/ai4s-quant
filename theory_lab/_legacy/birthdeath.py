"""生灭过程生成器与精确不变测度（ground truth）。

模型（含分布依赖项）:
    状态 i = 0, 1, 2, ...
    出生速率 b(i, mu) = b0 + b1 * i + b2 * m_mu
    死亡速率 a(i, mu) = a0 + a1 * i + a2 * m_mu
其中 m_mu 为经验分布的均值（分布依赖 / McKean-Vlasov 型耦合）。

无分布依赖（b2 = a2 = 0）时经典生灭过程有显式平稳分布:
    pi_n = pi_0 * prod_{k=0}^{n-1} b_k / a_{k+1}
本模块截断到 N 个状态，使尾部概率 < 1e-10，作为 ground truth。
"""
import numpy as np


class BirthDeathProcess:
    def __init__(self, N=96, b0=1.0, b1=0.85, a0=0.6, a1=1.0, b2=0.0, a2=0.0):
        """N: 截断状态数; 线性速率参数。"""
        self.N = N
        self.b0, self.b1, self.b2 = b0, b1, b2
        self.a0, self.a1, self.a2 = a0, a1, a2
        assert a1 > b1 > 0, "需要负漂移保证正常返（a1 > b1）"

    def rates(self, m_mu=0.0):
        """返回出生速率 b[0..N-2] 与死亡速率 a[1..N-1]（截断内）。"""
        i = np.arange(self.N)
        b = self.b0 + self.b1 * i + self.b2 * m_mu
        a = self.a0 + self.a1 * i + self.a2 * m_mu
        return b[:-1], a[1:]

    def generator(self, m_mu=0.0):
        """截断 Q-矩阵（生成元），行列和为零，最大特征值为 0。"""
        Q = np.zeros((self.N, self.N))
        b, a = self.rates(m_mu)
        for i in range(self.N - 1):
            Q[i, i + 1] = b[i]
            Q[i, i] -= b[i]
            Q[i + 1, i] = a[i]
            Q[i + 1, i + 1] -= a[i]
        return Q

    def stationary_exact(self):
        """无分布依赖情形的不变测度显式解（归一化），尾概率截断。"""
        b, a = self.rates(0.0)
        log_terms = np.log(b) - np.log(a)
        cum = np.concatenate([[0.0], np.cumsum(log_terms)])
        cum -= cum.max()
        pi = np.exp(cum)
        return pi / pi.sum()

    def stationary_mean(self):
        return float(np.arange(self.N) @ self.stationary_exact())

    def spectral_gap(self, m_mu=0.0):
        """截断生成元的谱隙：非零特征值中最大的 -Re(lambda)，即指数收敛率的精确值。"""
        Q = self.generator(m_mu)
        ev = np.linalg.eigvals(Q)
        re = np.sort(ev.real)[::-1]
        # ev[0] 应为 0（数值误差量级），取次大实部
        c = -re[1] if abs(re[0]) < 1e-8 else -re[0]
        return float(c)

    def sample_trajectory(self, T, rng, x0=None, m_mu=None, burn_in=0):
        """Gillespie 直接模拟，返回状态序列（长度 T+burn_in）。"""
        if m_mu is None:
            m_mu = self.stationary_mean()
        b, a = self.rates(m_mu)
        x = int(rng.integers(0, self.N)) if x0 is None else int(x0)
        out = np.empty(T + burn_in, dtype=np.int64)
        t = 0
        while t < T + burn_in:
            out[t] = x
            lam = b[x] if x < self.N - 1 else 0.0
            mu = a[x - 1] if x > 0 else 0.0
            tot = lam + mu
            if tot <= 0:
                t += 1
                continue
            r = rng.random() * tot
            if r < lam:
                x = min(x + 1, self.N - 1)
            else:
                x = max(x - 1, 0)
            t += 1
        return out[burn_in:]

    def mcmc(self, n_samples, rng, burn_in=2000, thin=5):
        """Metropolis（±1 提议）目标为截断平稳分布，返回样本序列。"""
        pi = self.stationary_exact()
        x = int(np.argmax(pi))
        keep = np.empty(n_samples * thin, dtype=np.int64)
        for t in range(n_samples * thin + burn_in):
            y = x + 1 if rng.random() < 0.5 else x - 1
            y = min(max(y, 0), self.N - 1)
            if rng.random() < min(1.0, pi[y] / pi[x]):
                x = y
            if t >= burn_in:
                keep[t - burn_in] = x
        return keep[::thin]
