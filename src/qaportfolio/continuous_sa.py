"""連續權重空間的模擬退火（scipy.optimize.dual_annealing）。

與 QUBO 不同，這裡直接在 [0,1]^n 上搜尋，再以 w / Σw 正規化，
因此可以直接最大化（非二次的）Sharpe ratio。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing


def _normalise(w_raw: np.ndarray) -> np.ndarray | None:
    s = w_raw.sum()
    return None if s <= 0 else w_raw / s


def _obj_variance(w_raw, cov):
    w = _normalise(w_raw)
    return 1e9 if w is None else float(np.sqrt(w @ cov @ w))


def _obj_neg_sharpe(w_raw, mu, cov):
    w = _normalise(w_raw)
    if w is None:
        return 1e9
    vol = np.sqrt(w @ cov @ w)
    return 1e9 if vol == 0 else float(-(mu @ w) / vol)


def solve(mu: pd.Series, cov: pd.DataFrame, kind: str, maxiter: int, initial_temp: float,
          seed: int) -> tuple[pd.Series, float]:
    m, C = mu.values, cov.loc[mu.index, mu.index].values
    bounds = [(0.0, 1.0)] * len(m)
    if kind == "GMVP":
        fun, args = _obj_variance, (C,)
    elif kind == "MSRP":
        fun, args = _obj_neg_sharpe, (m, C)
    else:
        raise ValueError(kind)
    t0 = time.perf_counter()
    res = dual_annealing(fun, bounds=bounds, args=args, maxiter=maxiter,
                         initial_temp=initial_temp, seed=seed)
    elapsed = time.perf_counter() - t0
    w = res.x / res.x.sum()
    w[w < 1e-3] = 0.0
    return pd.Series(w / w.sum(), index=mu.index), elapsed
