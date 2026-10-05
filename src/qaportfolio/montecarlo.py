"""事前（ex-ante）蒙地卡羅壓力測試：以決策日前一年的組合日報酬估計參數，
模擬持有期的報酬分布，計算 VaR / CVaR，並與實際實現報酬比對（是否穿越 VaR）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_terminal(daily_hist: pd.Series, horizon: int, n_paths: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu, sigma = daily_hist.mean(), daily_hist.std()
    sims = rng.normal(mu, sigma, size=(n_paths, horizon))
    return np.prod(1 + sims, axis=1) - 1


def var_cvar(terminal: np.ndarray, confidence: float) -> tuple[float, float, float]:
    var = np.percentile(terminal, (1 - confidence) * 100)
    return float(np.median(terminal)), float(var), float(terminal[terminal <= var].mean())


def ex_ante_stress(hist_returns: pd.DataFrame, weights: pd.DataFrame, horizon: int,
                   n_paths: int, confidence: float, seed: int) -> pd.DataFrame:
    """hist_returns：決策日前的個股日報酬；weights：股票 × 策略。"""
    rows = {}
    for s in weights.columns:
        w = weights[s].reindex(hist_returns.columns).fillna(0.0)
        port = hist_returns @ w
        med, var, cvar = var_cvar(simulate_terminal(port, horizon, n_paths, seed), confidence)
        rows[s] = {"Median": med, "VaR": var, "CVaR": cvar}
    return pd.DataFrame(rows).T
