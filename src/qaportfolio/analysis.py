"""彙整各季結果：回測、solver 品質、預測品質、集中度、oracle 距離、壓力測試。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import backtest, covariance, montecarlo
from .data import load_benchmark
from .pipeline import schedule_from_cfg

METHOD_ORDER = ["MVO", "MVO-round-P20", "MVO-round-P100", "SA-cont", "QUBO-P20", "QUBO-P100", "MVO-cap10"]


def split_name(s: str) -> tuple[str, str, str]:
    parts = s.split("|")
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else (s, "", "")


def load_quarter(cfg: dict, quarter: str) -> dict:
    d = cfg["paths"]["results"] / quarter
    return {
        "weights": pd.read_csv(d / "weights.csv", dtype={"stock_id": str}).set_index("stock_id"),
        "log": pd.read_csv(d / "solver_log.csv"),
        "mu": pd.read_csv(d / "mu.csv", dtype={"stock_id": str}).set_index("stock_id")["mu"],
        "forecast": json.loads((d / "forecast_metrics.json").read_text(encoding="utf-8")),
        "preds": pd.read_csv(d / "forecast_predictions.csv", dtype={"stock_id": str}, parse_dates=["date"]),
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
    }


def run(cfg: dict, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    schedule = schedule_from_cfg(cfg)
    Q = {w.quarter: load_quarter(cfg, w.quarter) for w in schedule}
    bt = cfg["backtest"]

    # --- 回測（oracle 另外跑，只作參考） ---
    W = {q: v["weights"] for q, v in Q.items()}
    daily, quarterly = backtest.run_walkforward(prices, schedule, W, bt["fee_rate"], bt["tax_rate"])
    daily = daily.join(backtest.benchmark_returns(load_benchmark(cfg["paths"]["benchmark"]), schedule))
    perf = backtest.performance_table(daily)
    bench_q = []
    bench = load_benchmark(cfg["paths"]["benchmark"])
    for w in schedule:
        px = bench.loc[w.announcement:w.hold_end]
        bench_q.append({"quarter": w.quarter, "strategy": "0050 ETF", "return": px.iloc[-1] / px.iloc[0] - 1})
    quarterly = pd.concat([quarterly, pd.DataFrame(bench_q)], ignore_index=True)

    meta = pd.DataFrame([split_name(s) for s in perf.index], index=perf.index,
                        columns=["method", "estimator", "portfolio"])
    perf = meta.join(perf)
    avg_turn = quarterly.groupby("strategy")[["turnover", "cost"]].mean()
    perf = perf.join(avg_turn.rename(columns={"turnover": "Avg Turnover", "cost": "Avg Cost"}))
    mkt = daily["0050 ETF"]
    perf["Beta (0050)"] = daily.apply(lambda r: r.cov(mkt) / mkt.var())

    # 依方法彙總（跨 5 種共變異數估計式的中位數），排除 oracle
    core = perf[perf["method"].isin(METHOD_ORDER)]
    by_method = (core.groupby(["portfolio", "method"])[["Ann. Return", "Ann. Vol", "Sharpe",
                                                        "Max Drawdown", "Beta (0050)", "Avg Turnover"]]
                 .median().reindex(pd.MultiIndex.from_product([["GMVP", "MSRP"], METHOD_ORDER])))

    # 每季勝率：同一估計式、同一組合類型下，各方法的季報酬排名
    qr = quarterly[quarterly["strategy"].str.count(r"\|") == 2].copy()
    qr[["method", "estimator", "portfolio"]] = pd.DataFrame(qr["strategy"].map(split_name).tolist(), index=qr.index)
    qr = qr[qr["method"].isin(METHOD_ORDER)]
    qr["rank"] = qr.groupby(["quarter", "estimator", "portfolio"])["return"].rank(ascending=False)
    rank_table = qr.pivot_table(index="method", columns="portfolio", values="rank", aggfunc="mean").reindex(METHOD_ORDER)

    # --- solver 品質 ---
    log = pd.concat([v["log"] for v in Q.values()], ignore_index=True)
    mv = log[log["problem"] == "MV"]
    quality_mv = mv.groupby(["method", "gamma"])[["ce_loss", "round_ce_loss", "feasible_rate", "time"]].mean()
    g = log[(log["problem"] == "GMVP") & log["method"].str.startswith("QUBO")]
    quality_gmvp = g.groupby("method")[["vol_ratio", "round_vol_ratio", "feasible_rate", "time"]].mean()
    timing = log.dropna(subset=["time"]).groupby(["method", "problem"])["time"].median().unstack()
    sel = log[log["problem"] == "MSRP-select"][["quarter", "estimator", "method", "gamma"]]
    sel = sel.merge(mv[["quarter", "estimator", "method", "gamma", "repaired"]],
                    on=["quarter", "estimator", "method", "gamma"], how="left")
    sel["gamma"] = sel["gamma"].fillna(np.inf)          # inf = 選到 GMVP 解
    msrp_gamma = sel.groupby("method")["gamma"].value_counts().unstack(fill_value=0)
    msrp_gamma["repaired_share"] = sel.groupby("method")["repaired"].apply(
        lambda r: r.eq(True).mean())

    # --- 預測品質 ---
    fc = pd.DataFrame({q: v["forecast"] for q, v in Q.items()}).T
    ic_at_decision = {}
    for w in schedule:
        p = Q[w.quarter]["preds"]
        mu = Q[w.quarter]["mu"]
        px = prices.loc[w.announcement:w.hold_end, mu.index].ffill()
        realized = px.iloc[-1] / px.iloc[0] - 1
        ic_at_decision[w.quarter] = mu.corr(realized, method="spearman")
    fc["RankIC_decision_vs_holding"] = pd.Series(ic_at_decision)

    # --- 集中度與 oracle 距離 ---
    conc_rows, dist_rows = [], []
    for q, v in Q.items():
        Wq = v["weights"]
        for s in Wq.columns:
            m, e, port = split_name(s)
            w = Wq[s]
            conc_rows.append(dict(quarter=q, method=m, estimator=e, portfolio=port,
                                  eff_n=1 / (w ** 2).sum(), n_holdings=int((w > 1e-6).sum()), max_w=w.max()))
            if m in METHOD_ORDER:
                o = Wq[f"Oracle|Realized|{port}"]
                dist_rows.append(dict(quarter=q, method=m, estimator=e, portfolio=port,
                                      l1_to_oracle=(w - o).abs().sum()))
    conc = pd.DataFrame(conc_rows)
    concentration = conc.groupby(["portfolio", "method"])[["eff_n", "n_holdings", "max_w"]].mean()
    oracle_dist = pd.DataFrame(dist_rows).groupby(["portfolio", "method"])["l1_to_oracle"].mean().unstack(0)

    # --- 事前壓力測試（主估計式）與實現報酬比對 ---
    prim = covariance.ESTIMATOR_LABELS[cfg["covariance"]["primary"]]
    mc = cfg["monte_carlo"]
    stress_rows = []
    for w in schedule:
        Wq = Q[w.quarter]["weights"]
        cols = [c for c in Wq.columns if f"|{prim}|" in c or c == "EqualWeight"]
        hist = covariance.returns_window(prices[Wq.index], w.announcement, cfg["covariance"]["lookback_days"])
        horizon = len(prices.loc[w.announcement:w.hold_end]) - 1
        st = montecarlo.ex_ante_stress(hist, Wq[cols], horizon, mc["n_paths"], mc["confidence"], cfg["seed"])
        realized = quarterly[quarterly["quarter"] == w.quarter].set_index("strategy")["return"]
        st["Realized"] = realized.reindex(st.index)
        st["Breach"] = st["Realized"] < st["VaR"]
        st.insert(0, "quarter", w.quarter)
        stress_rows.append(st.rename_axis("strategy").reset_index())
    stress = pd.concat(stress_rows, ignore_index=True)

    return dict(daily=daily, performance=perf, by_method=by_method, quarterly=quarterly,
                rank_table=rank_table, solver_log=log, quality_mv=quality_mv,
                quality_gmvp=quality_gmvp, timing=timing, msrp_gamma=msrp_gamma,
                forecast=fc, concentration=concentration, concentration_raw=conc,
                oracle_distance=oracle_dist, stress=stress)
