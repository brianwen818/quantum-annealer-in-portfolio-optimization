"""共變異數矩陣估計（季化單位，frequency=63）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import risk_models

ESTIMATOR_LABELS = {
    "sample": "Sample",
    "ewma": "EWMA",
    "lw_identity": "LW-Identity",
    "lw_const_corr": "LW-ConstCorr",
    "lw_single_factor": "LW-SingleFactor",
}


def returns_window(prices: pd.DataFrame, end: pd.Timestamp, n_days: int) -> pd.DataFrame:
    """end（含）往前 n_days 個交易日的日報酬。個別缺值（停牌）視為 0 報酬。"""
    rets = prices.loc[:end].pct_change(fill_method=None).iloc[1:]
    rets = rets.dropna(how="all").iloc[-n_days:]
    return rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def estimate(returns: pd.DataFrame, method: str, frequency: int = 63, ewma_span: int = 180) -> pd.DataFrame:
    kw = dict(returns_data=True, frequency=frequency)
    if method == "sample":
        cov = risk_models.sample_cov(returns, **kw)
    elif method == "ewma":
        cov = risk_models.exp_cov(returns, span=ewma_span, **kw)
    elif method == "lw_identity":
        cov = risk_models.CovarianceShrinkage(returns, **kw).ledoit_wolf()
    elif method == "lw_const_corr":
        cov = risk_models.risk_matrix(returns, method="ledoit_wolf_constant_correlation", **kw)
    elif method == "lw_single_factor":
        cov = risk_models.risk_matrix(returns, method="ledoit_wolf_single_factor", **kw)
    else:
        raise ValueError(method)
    return risk_models.fix_nonpositive_semidefinite(cov, fix_method="spectral")


def condition_number(cov: pd.DataFrame) -> float:
    eig = np.linalg.eigvalsh(cov.values)
    return float(eig.max() / max(eig.min(), 1e-16))
