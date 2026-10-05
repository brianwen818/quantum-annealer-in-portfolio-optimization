"""彙整 results/<quarter>/ 的輸出，產生 results/summary/*.csv 與 results/figures/*.png。

用法：python scripts/analyze.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from qaportfolio import analysis, plotting  # noqa: E402
from qaportfolio.config import load_config  # noqa: E402
from qaportfolio.covariance import ESTIMATOR_LABELS  # noqa: E402
from qaportfolio.pipeline import load_prices, schedule_from_cfg  # noqa: E402


def main() -> None:
    cfg = load_config()
    prices = load_prices(cfg)
    res = analysis.run(cfg, prices)
    schedule = schedule_from_cfg(cfg)

    out = cfg["paths"]["results"] / "summary"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["performance", "by_method", "quarterly", "rank_table", "solver_log", "quality_mv",
                 "quality_gmvp", "timing", "msrp_gamma", "forecast", "concentration",
                 "oracle_distance", "stress"]:
        df = res[name]
        df.to_csv(out / f"{name}.csv", index=not isinstance(df.index, pd.RangeIndex), float_format="%.6g")
    res["daily"].to_csv(out / "daily_returns.csv", float_format="%.8g")

    figs = cfg["paths"]["results"] / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    prim = ESTIMATOR_LABELS[cfg["covariance"]["primary"]]
    first_q = schedule[0].quarter
    W0 = analysis.load_quarter(cfg, first_q)["weights"]
    preds = pd.concat([analysis.load_quarter(cfg, w.quarter)["preds"] for w in schedule])

    for fname, fig in {
        "nav_walkforward.png": plotting.nav_chart(res["daily"], prim, schedule),
        "solver_quality.png": plotting.solver_quality_chart(res["quality_mv"]),
        f"weights_{first_q}.png": plotting.weights_chart(W0, prim),
        "concentration.png": plotting.concentration_chart(res["concentration_raw"]),
        "forecast_scatter.png": plotting.forecast_scatter(preds),
        "quarterly_returns.png": plotting.quarterly_heatmap(res["quarterly"], prim),
    }.items():
        fig.savefig(figs / fname, bbox_inches="tight", dpi=130)

    pd.set_option("display.width", 200)
    print(res["by_method"].round(4))
    print(res["rank_table"].round(2))
    print(res["quality_gmvp"].round(4))
    print(res["forecast"].round(4))


if __name__ == "__main__":
    main()
