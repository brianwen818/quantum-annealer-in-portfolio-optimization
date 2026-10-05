"""LSTM 的特徵工程與標籤（未來一季報酬）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "return_fwd_1q"

FEATURES = [
    "Trading_Volume", "Trading_money", "open", "max", "min", "close", "spread",
    "Trading_turnover", "daily_return", "MA_20", "MA_60", "Bias_20", "Bias_60",
    "ROC_20", "ROC_60", "Vol_60", "Vol_MA_20", "Volume_Ratio_20", "Price_Position_60",
]


def add_features(df: pd.DataFrame, horizon: int = 63) -> pd.DataFrame:
    """所有特徵僅使用當日（含）以前的資料；標籤為 t → t+horizon 的報酬。"""
    df = df.sort_values("date").reset_index(drop=True).copy()
    c = df["close"]
    df["daily_return"] = c.pct_change(fill_method=None)
    df[TARGET] = c.shift(-horizon) / c - 1

    df["MA_20"] = c.rolling(20).mean()
    df["MA_60"] = c.rolling(60).mean()
    df["Bias_20"] = (c - df["MA_20"]) / df["MA_20"]
    df["Bias_60"] = (c - df["MA_60"]) / df["MA_60"]
    df["ROC_20"] = c.pct_change(20, fill_method=None)
    df["ROC_60"] = c.pct_change(60, fill_method=None)
    df["Vol_60"] = df["daily_return"].rolling(60).std() * np.sqrt(252)
    df["Vol_MA_20"] = df["Trading_Volume"].rolling(20).mean()
    df["Volume_Ratio_20"] = df["Trading_Volume"] / (df["Vol_MA_20"] + 1e-8)
    hi, lo = df["max"].rolling(60).max(), df["min"].rolling(60).min()
    df["Price_Position_60"] = (c - lo) / (hi - lo + 1e-8)

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).ffill()
    return df
