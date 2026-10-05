"""重新建立 data/ 的工具腳本（repo 已附資料，一般不需要執行）。

* 個股：FinMind `TaiwanStockPriceAdj`（還原股價）。此資料集需 FinMind 付費等級，
  請設定環境變數 FINMIND_TOKEN。
* 0050 ETF 基準：Yahoo Finance `0050.TW`（auto_adjust 還原權息與 2025/06 的 1 拆 4）。
* 0050 成分股持股明細（data/meta/Holding_0050.xlsx）來自 TEJ，需另行由 TEJ 匯出。

用法：
    python scripts/download_data.py --stocks 2330 2317 --start 1994-01-01
    python scripts/download_data.py --benchmark
    python scripts/download_data.py --universe    # 依 config 季度的成分股重抓全部個股
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from qaportfolio.config import load_config  # noqa: E402
from qaportfolio.data import constituents, load_holdings  # noqa: E402


def download_stocks(ids: list[str], start: str, end: str, out_dir: Path) -> None:
    from FinMind.data import DataLoader

    dl = DataLoader()
    token = os.environ.get("FINMIND_TOKEN")
    if token:
        dl.login_by_token(api_token=token)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        df = dl.taiwan_stock_daily_adj(stock_id=sid, start_date=start, end_date=end)
        df.reset_index(drop=True).to_feather(out_dir / f"{sid}.feather")
        print(sid, len(df))


def download_benchmark(start: str, end: str, out: Path) -> None:
    import yfinance as yf

    df = yf.download("0050.TW", start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "date", "Close": "close", "Open": "open",
                                          "High": "max", "Low": "min", "Volume": "Trading_Volume"})
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["stock_id"] = "0050"
    df[["date", "stock_id", "Trading_Volume", "open", "max", "min", "close"]].to_feather(out)
    print("0050", len(df))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="*", default=[])
    ap.add_argument("--universe", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--start", default="1994-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = ap.parse_args()
    cfg = load_config()

    ids = list(args.stocks)
    if args.universe:
        h = load_holdings(cfg["paths"]["holdings"])
        ids += sorted({c for q in cfg["quarters"] for c in constituents(h, q)})
    if ids:
        download_stocks(sorted(set(ids)), args.start, args.end, cfg["paths"]["stocks"])
    if args.benchmark:
        download_benchmark("2003-06-01", args.end, cfg["paths"]["benchmark"])


if __name__ == "__main__":
    main()
