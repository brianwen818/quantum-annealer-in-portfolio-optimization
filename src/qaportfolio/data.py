"""資料讀取：個股還原股價、0050 成分股、交易日曆、0050 ETF 基準。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_trading_dates(path: Path) -> pd.DatetimeIndex:
    dates = pd.to_datetime(pd.read_feather(path)["date"])
    return pd.DatetimeIndex(dates.drop_duplicates().sort_values())


def load_holdings(path: Path) -> pd.DataFrame:
    """TEJ 0050 持股明細。`年月` 例：'2025/03'。"""
    return pd.read_excel(path, dtype={"標的碼": str})


def constituents(holdings: pd.DataFrame, quarter: str) -> list[str]:
    """指定季度（'2025_03'）的 0050 成分股（排除產業／指數彙總列等非數字代碼）。"""
    ym = quarter.replace("_", "/")
    codes = holdings.loc[holdings["年月"] == ym, "標的碼"].astype(str).unique()
    return sorted(c for c in codes if c.isdigit())


def stock_names(holdings: pd.DataFrame) -> dict[str, str]:
    h = holdings.sort_values("年月")
    return dict(zip(h["標的碼"].astype(str), h["標的名稱"]))


def load_stock(stocks_dir: Path, stock_id: str) -> pd.DataFrame:
    df = pd.read_feather(stocks_dir / f"{stock_id}.feather")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def load_stocks(stocks_dir: Path, stock_ids: list[str]) -> dict[str, pd.DataFrame]:
    return {sid: load_stock(stocks_dir, sid) for sid in stock_ids}


def load_benchmark(path: Path) -> pd.Series:
    df = pd.read_feather(path)
    return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]), name="0050")


def close_panel(stocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """日期 × 股票 的還原收盤價矩陣。"""
    return pd.DataFrame({sid: df.set_index("date")["close"] for sid, df in stocks.items()}).sort_index()
