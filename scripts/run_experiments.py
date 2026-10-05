"""執行 walk-forward 實驗（每季：LSTM → 共變異數 → MVO / SA / QUBO）。

用法：
    python scripts/run_experiments.py                    # 跑設定檔中的全部季度
    python scripts/run_experiments.py --quarters 2025_03 # 只跑一季
    python scripts/run_experiments.py --force-forecast   # 重新訓練 LSTM（否則沿用 results/<q>/mu.csv）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaportfolio.config import load_config, set_seed  # noqa: E402
from qaportfolio.pipeline import (load_prices, refresh_convex, run_quarter,  # noqa: E402
                                  schedule_from_cfg, step_forecast, universe_for)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--quarters", nargs="*", default=None)
    ap.add_argument("--force-forecast", action="store_true")
    ap.add_argument("--forecast-only", action="store_true", help="只訓練 LSTM、輸出 mu.csv")
    ap.add_argument("--refresh-convex", action="store_true",
                    help="只重算顯式正則化的凸最佳化對照組（沿用既有 mu／共變異數／退火結果）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    prices = load_prices(cfg)
    for win in schedule_from_cfg(cfg):
        if args.quarters and win.quarter not in args.quarters:
            continue
        t0 = time.time()
        print(f"=== {win.quarter}: decide {win.announcement.date()} → hold to {win.hold_end.date()} ===", flush=True)
        if args.refresh_convex:
            refresh_convex(cfg, win)
        elif args.forecast_only:
            _, stocks = universe_for(cfg, win)
            step_forecast(cfg, win, stocks, force=args.force_forecast)
        else:
            run_quarter(cfg, win, prices, force_forecast=args.force_forecast)
        print(f"=== {win.quarter} done in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
