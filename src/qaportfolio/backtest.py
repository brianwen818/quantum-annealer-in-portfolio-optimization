"""Walk-forward 回測與績效指標。

* 每季於公告日收盤依目標權重建倉，季內買進持有（權重隨價格漂移），
  下一季公告日收盤再平衡。
* 再平衡成本：買進收手續費，賣出收手續費＋證交稅，以周轉率計算後於建倉時扣除。
* 0050 基準為 ETF 買進持有（不計成本）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def holding_path(prices: pd.DataFrame, weights: pd.Series, start: pd.Timestamp,
                 end: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """回傳 (start 之後每日的組合淨值（start = 1），期末漂移後權重)。"""
    w = weights[weights > 0]
    px = prices.loc[start:end, w.index].ffill()
    rel = px / px.iloc[0]
    value = rel.mul(w, axis=1).sum(axis=1)
    end_w = rel.iloc[-1] * w / value.iloc[-1]
    return value, end_w


def rebalance_cost(prev: pd.Series | None, target: pd.Series, fee: float, tax: float) -> tuple[float, float]:
    """回傳 (單邊周轉率, 交易成本占組合價值比例)。"""
    prev = pd.Series(dtype=float) if prev is None else prev
    idx = prev.index.union(target.index)
    diff = target.reindex(idx, fill_value=0) - prev.reindex(idx, fill_value=0)
    buys, sells = diff.clip(lower=0).sum(), (-diff).clip(lower=0).sum()
    return float(max(buys, sells)), float(buys * fee + sells * (fee + tax))


def run_walkforward(prices: pd.DataFrame, schedule, weights_by_quarter: dict[str, pd.DataFrame],
                    fee: float, tax: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """weights_by_quarter[q]：index = 股票、columns = 策略。

    回傳 (每日報酬 DataFrame[日期 × 策略], 每季紀錄 DataFrame)。
    """
    strategies = list(next(iter(weights_by_quarter.values())).columns)
    daily, records = {s: [] for s in strategies}, []
    prev_w = {s: None for s in strategies}

    for win in schedule:
        W = weights_by_quarter[win.quarter]
        for s in strategies:
            target = W[s].fillna(0.0)
            target = target[target > 0]
            turnover, cost = rebalance_cost(prev_w[s], target, fee, tax)
            value, end_w = holding_path(prices, target, win.announcement, win.hold_end)
            value = value * (1 - cost)
            rets = value.pct_change().iloc[1:]
            rets.iloc[0] = value.iloc[1] / 1.0 - 1          # 第一天報酬含建倉成本
            daily[s].append(rets)
            prev_w[s] = end_w
            records.append({"quarter": win.quarter, "strategy": s,
                            "return": float(value.iloc[-1] - 1),
                            "turnover": turnover, "cost": cost,
                            "n_holdings": int((target > 1e-6).sum()),
                            "max_weight": float(target.max())})
    daily_df = pd.DataFrame({s: pd.concat(v) for s, v in daily.items()})
    return daily_df, pd.DataFrame(records)


def benchmark_returns(bench: pd.Series, schedule) -> pd.Series:
    start, end = schedule[0].announcement, schedule[-1].hold_end
    px = bench.loc[start:end]
    return px.pct_change().iloc[1:].rename("0050 ETF")


def performance(returns: pd.Series) -> dict:
    r = returns.dropna()
    nav = (1 + r).cumprod()
    total = nav.iloc[-1] - 1
    ann_ret = nav.iloc[-1] ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    mdd = (nav / nav.cummax() - 1).min()
    return {
        "Total Return": total,
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan,
        "Max Drawdown": mdd,
        "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
    }


def performance_table(daily: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: performance(daily[c]) for c in daily.columns}).T
