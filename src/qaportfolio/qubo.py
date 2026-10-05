r"""把 long-only 均值—變異數問題寫成 QUBO。

離散化
------
把資金切成 P 份（lot），x_i ∈ {0,…,P} 為第 i 檔股票分到的份數，w_i = x_i / P。
每個整數以「有界二進位展開」表示：x_i = Σ_k c_k b_{ik}，b_{ik} ∈ {0,1}。
例如 P = 20 → c = (1, 2, 4, 8, 5)；P = 100 → c = (1, 2, 4, 8, 16, 32, 37)。

目標函數（最小化）
------------------
    MV(γ):  (γ / 2P²) xᵀΣx − (1/P) μᵀx      ⇔  (γ/2) wᵀΣw − μᵀw
    GMVP :  (1 / P²)  xᵀΣx                   ⇔  wᵀΣw

預算限制 Σx_i = P 以 penalty λ(Σx_i − P)² 加入。λ 取為
「每違反一個 lot 所能換到的最大目標改善量」L 的 penalty_factor 倍；
因為整數違反量 d 的 penalty λd² ≥ λ|d| > L|d|，最低能量解必然可行。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import dimod
import numpy as np
import pandas as pd

from .samplers import sample_bqm


def binary_coefficients(upper: int) -> np.ndarray:
    coefs, total, k = [], 0, 1
    while total + k <= upper:
        coefs.append(k)
        total += k
        k *= 2
    if total < upper:
        coefs.append(upper - total)
    return np.array(coefs, dtype=float)


@dataclass
class QuboProblem:
    bqm: dimod.BinaryQuadraticModel
    coefs: np.ndarray
    n_assets: int
    precision: int
    lagrange: float


def build_qubo(mu: np.ndarray, cov: np.ndarray, precision: int, gamma: float | None,
               penalty_factor: float = 2.0) -> QuboProblem:
    """gamma=None 代表 GMVP（只最小化變異數）。"""
    n, P = len(mu), precision
    c = binary_coefficients(P)
    K = len(c)
    A = np.kron(np.eye(n), c)                     # (n, n·K)：x = A b
    a = A.sum(axis=0)                             # 每個 bit 代表的 lot 數

    if gamma is None:
        quad_scale, lin = 1.0 / P**2, np.zeros(n)
        L = 2.0 / P * np.abs(cov).max() + np.diag(cov).max() / P**2
    else:
        quad_scale, lin = gamma / (2 * P**2), -mu / P
        L = gamma / P * np.abs(cov).max() + np.abs(mu).max() / P + gamma * np.diag(cov).max() / (2 * P**2)
    lam = penalty_factor * L

    M = quad_scale * A.T @ cov @ A + lam * np.outer(a, a)
    linear = A.T @ lin - 2 * lam * P * a + np.diag(M)
    iu = np.triu_indices(n * K, 1)
    bqm = dimod.BinaryQuadraticModel.from_numpy_vectors(
        linear, (iu[0], iu[1], 2 * M[iu]), lam * P**2, dimod.BINARY)
    return QuboProblem(bqm, c, n, P, lam)


@dataclass
class QuboSolution:
    weights: np.ndarray
    feasible_rate: float     # 所有 reads 中滿足預算限制的比例
    repaired: bool           # 沒有任何可行解時，以等比例縮放＋最大餘數法修復
    sample_time: float       # 只計 sampler 的時間（不含建模）
    build_time: float


def decode(problem: QuboProblem, sampleset: dimod.SampleSet) -> tuple[np.ndarray, float, bool]:
    rec = sampleset.record
    order = np.argsort(rec.energy)
    bits = rec.sample[order].reshape(len(order), problem.n_assets, len(problem.coefs))
    x = bits @ problem.coefs                      # (reads, n)
    ok = x.sum(axis=1) == problem.precision
    rate = float(np.repeat(ok, rec.num_occurrences[order]).mean())
    if ok.any():
        return x[np.argmax(ok)] / problem.precision, rate, False
    best = x[0]
    if best.sum() == 0:
        return np.full(problem.n_assets, 1 / problem.n_assets), rate, True
    raw = best / best.sum() * problem.precision
    lots = np.floor(raw)
    lots[np.argsort(-(raw - lots))[: int(problem.precision - lots.sum())]] += 1
    return lots / problem.precision, rate, True


def solve(mu: pd.Series, cov: pd.DataFrame, precision: int, gamma: float | None,
          sampler: str, sampler_params: dict, penalty_factor: float, seed: int) -> QuboSolution:
    t0 = time.perf_counter()
    prob = build_qubo(mu.values, cov.loc[mu.index, mu.index].values, precision, gamma, penalty_factor)
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    ss = sample_bqm(prob.bqm, sampler, sampler_params, seed)
    elapsed = time.perf_counter() - t0
    w, rate, repaired = decode(prob, ss)
    return QuboSolution(w, rate, repaired, elapsed, build)
