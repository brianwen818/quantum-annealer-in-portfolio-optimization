"""古典凸最佳化（PyPortfolioOpt）與共用的目標函數。

統一使用 long-only、權重和為 1 的限制。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier


def mv_objective(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, gamma: float) -> float:
    """最小化形式：(gamma/2) w'Σw - mu'w（= 負的均值—變異數效用）。"""
    return float(0.5 * gamma * w @ cov @ w - mu @ w)


def sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    vol = np.sqrt(max(w @ cov @ w, 1e-18))
    return float(mu @ w / vol)


def _clean(w: dict, index) -> pd.Series:
    s = pd.Series(w).reindex(index).fillna(0.0).clip(lower=0)
    s[s < 1e-6] = 0.0
    return s / s.sum()


def gmvp(mu: pd.Series, cov: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, max_weight))
    return _clean(ef.min_volatility(), mu.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    """Rf = 0。若所有預期報酬皆 <= 0，最大 Sharpe 無定義，退回 GMVP。"""
    if (mu <= 0).all():
        return gmvp(mu, cov, max_weight)
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, max_weight))
    return _clean(ef.max_sharpe(risk_free_rate=0.0), mu.index)


def max_utility(mu: pd.Series, cov: pd.DataFrame, gamma: float) -> pd.Series:
    ef = EfficientFrontier(mu, cov)
    return _clean(ef.max_quadratic_utility(risk_aversion=gamma), mu.index)


def round_to_lots(w: pd.Series, precision: int) -> pd.Series:
    """最大餘數法把連續權重四捨五入到 1/precision 的格點（古典離散化基準）。"""
    raw = w.values * precision
    lots = np.floor(raw).astype(int)
    short = precision - lots.sum()
    order = np.argsort(-(raw - lots))
    lots[order[:short]] += 1
    return pd.Series(lots / precision, index=w.index)
