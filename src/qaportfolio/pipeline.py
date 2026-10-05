"""單季完整流程：成分股 → LSTM 預期報酬 → 共變異數 → 各 solver 求解 → 輸出權重與求解紀錄。

輸出到 results/<quarter>/：
  mu.csv, forecast_predictions.csv, forecast_metrics.json, training_history.csv,
  cov_<estimator>.csv, weights.csv（股票 × 策略）, solver_log.csv, meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import continuous_sa, covariance, mvo, qubo
from .data import constituents, load_holdings, load_stocks, load_trading_dates, close_panel
from .schedule import QuarterWindow, build_schedule

PORTS = ("GMVP", "MSRP")


def strategy_name(method: str, estimator: str, port: str) -> str:
    return f"{method}|{covariance.ESTIMATOR_LABELS.get(estimator, estimator)}|{port}"


def quarter_dir(cfg: dict, quarter: str) -> Path:
    d = cfg["paths"]["results"] / quarter
    d.mkdir(parents=True, exist_ok=True)
    return d


def schedule_from_cfg(cfg: dict) -> list[QuarterWindow]:
    td = load_trading_dates(cfg["paths"]["trading_dates"])
    return build_schedule(cfg["quarters"], cfg["next_announcement_after_last"], td)


def universe_for(cfg: dict, win: QuarterWindow) -> tuple[list[str], dict[str, pd.DataFrame]]:
    holdings = load_holdings(cfg["paths"]["holdings"])
    codes = constituents(holdings, win.quarter)
    stocks = load_stocks(cfg["paths"]["stocks"], codes)
    keep = [s for s in codes
            if (stocks[s]["date"] <= win.announcement).sum() >= cfg["universe"]["min_history_days"]]
    return keep, {s: stocks[s] for s in keep}


# ---------------------------------------------------------------------------
# Step 1：預期報酬
# ---------------------------------------------------------------------------
def step_forecast(cfg: dict, win: QuarterWindow, stocks: dict, force: bool = False) -> pd.Series:
    out = quarter_dir(cfg, win.quarter)
    if (out / "mu.csv").exists() and not force:
        return pd.read_csv(out / "mu.csv", dtype={"stock_id": str}).set_index("stock_id")["mu"]

    from .forecast_lstm import forecast_metrics, train_and_forecast

    res = train_and_forecast(stocks, win.announcement, cfg["lstm"], cfg["seed"], verbose=2)
    res.mu.rename_axis("stock_id").to_csv(out / "mu.csv")
    res.predictions.to_csv(out / "forecast_predictions.csv", index=False)
    res.history.to_csv(out / "training_history.csv", index_label="epoch")
    m = forecast_metrics(res.predictions)
    m.update(n_train_sequences=res.n_train, n_val_sequences=res.n_val,
             best_epoch=int(np.argmin(res.history["val_loss"])) + 1)
    (out / "forecast_metrics.json").write_text(json.dumps(m, indent=2, default=float), encoding="utf-8")
    return res.mu


# ---------------------------------------------------------------------------
# Step 2：共變異數
# ---------------------------------------------------------------------------
def step_covariances(cfg: dict, win: QuarterWindow, prices: pd.DataFrame,
                     tickers: list[str]) -> dict[str, pd.DataFrame]:
    c = cfg["covariance"]
    rets = covariance.returns_window(prices[tickers], win.announcement, c["lookback_days"])
    out = quarter_dir(cfg, win.quarter)
    covs = {}
    for est in c["estimators"]:
        cov = covariance.estimate(rets, est, c["frequency"], c["ewma_span"]).loc[tickers, tickers]
        cov.to_csv(out / f"cov_{est}.csv")
        covs[est] = cov
    return covs


def oracle_inputs(prices: pd.DataFrame, win: QuarterWindow, tickers: list[str],
                  frequency: int) -> tuple[pd.Series, pd.DataFrame]:
    """事後才知道的「真實」報酬與共變異數（只用於診斷，不參與策略比較）。"""
    px = prices.loc[win.announcement:win.hold_end, tickers].ffill()
    mu = (px.iloc[-1] / px.iloc[0] - 1).rename("mu")
    rets = px.pct_change().iloc[1:].fillna(0.0)
    return mu, rets.cov() * frequency


# ---------------------------------------------------------------------------
# Step 3：求解（平行化）
# ---------------------------------------------------------------------------
def _qubo_task(mu, cov, P, gamma, sampler, params, penalty, seed):
    s = qubo.solve(mu, cov, P, gamma, sampler, params, penalty, seed)
    return pd.Series(s.weights, index=mu.index), s


def _sa_task(mu, cov, kind, sa_cfg, seed):
    return continuous_sa.solve(mu, cov, kind, sa_cfg["maxiter"], sa_cfg["initial_temp"], seed)


def convex_regularised(cfg: dict, mu: pd.Series, covs: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """顯式正則化的古典對照組：MVO 加單一權重上限（檢驗 QUBO 的分散效果能否被簡單限制複製）。"""
    cap = cfg["max_weight_cap"]
    name = f"MVO-cap{int(round(cap * 100))}"
    out = {}
    for est, cov in covs.items():
        out[strategy_name(name, est, "GMVP")] = mvo.gmvp(mu, cov, cap)
        out[strategy_name(name, est, "MSRP")] = mvo.max_sharpe(mu, cov, cap)
    return out


def refresh_convex(cfg: dict, win: QuarterWindow) -> None:
    """不重跑 LSTM／退火，只以既有的 mu 與共變異數重新計算顯式正則化對照組並併入 weights.csv。"""
    out = quarter_dir(cfg, win.quarter)
    mu = pd.read_csv(out / "mu.csv", dtype={"stock_id": str}).set_index("stock_id")["mu"]
    W = pd.read_csv(out / "weights.csv", dtype={"stock_id": str}).set_index("stock_id")
    mu = mu.loc[W.index]
    covs = {}
    for est in cfg["covariance"]["estimators"]:
        c = pd.read_csv(out / f"cov_{est}.csv", index_col=0)
        c.index, c.columns = c.index.astype(str), c.columns.astype(str)
        covs[est] = c.loc[mu.index, mu.index]
    for k, v in convex_regularised(cfg, mu, covs).items():
        W[k] = v.reindex(W.index).fillna(0.0)
    W.to_csv(out / "weights.csv")


def step_solve(cfg: dict, win: QuarterWindow, mu: pd.Series,
               covs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed, gammas, qcfg = cfg["seed"], cfg["gamma_grid"], cfg["qubo"]
    weights: dict[str, pd.Series] = {}
    log: list[dict] = []

    # --- 排程所有耗時任務 ---
    jobs, keys = [], []
    for est, cov in covs.items():
        for kind in PORTS:
            jobs.append(delayed(_sa_task)(mu, cov, kind, cfg["continuous_sa"], seed))
            keys.append(("SA", est, kind, None))
        for pname, p in qcfg["precisions"].items():
            params = {"num_reads": p["num_reads"], "num_sweeps": p["num_sweeps"]}
            for g in [None] + gammas:
                jobs.append(delayed(_qubo_task)(mu, cov, p["precision"], g, qcfg["sampler"],
                                                params, qcfg["penalty_factor"], seed))
                keys.append((pname, est, "GMVP" if g is None else "MV", g))
    results = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(jobs)
    done = dict(zip(keys, results))

    for est, cov in covs.items():
        m, C = mu.values, cov.loc[mu.index, mu.index].values

        # 古典凸最佳化（精確解）
        w_g, w_s = mvo.gmvp(mu, cov), mvo.max_sharpe(mu, cov)
        weights[strategy_name("MVO", est, "GMVP")] = w_g
        weights[strategy_name("MVO", est, "MSRP")] = w_s
        cont_mv = {g: mvo.max_utility(mu, cov, g) for g in gammas}
        var_star = float(w_g @ C @ w_g)

        # 連續 SA
        for kind in PORTS:
            w, t = done[("SA", est, kind, None)]
            weights[strategy_name("SA-cont", est, kind)] = w
            log.append(dict(estimator=est, method="SA-cont", problem=kind, time=t,
                            sharpe=mvo.sharpe(w.values, m, C), vol=np.sqrt(w @ C @ w)))

        for pname, p in qcfg["precisions"].items():
            P = p["precision"]
            # 古典離散化基準：把凸最佳解四捨五入到 1/P 格點
            weights[strategy_name(f"MVO-round-{pname}", est, "GMVP")] = mvo.round_to_lots(w_g, P)
            weights[strategy_name(f"MVO-round-{pname}", est, "MSRP")] = mvo.round_to_lots(w_s, P)

            # QUBO：GMVP
            w, s = done[(pname, est, "GMVP", None)]
            wr = mvo.round_to_lots(w_g, P).values
            weights[strategy_name(f"QUBO-{pname}", est, "GMVP")] = w
            log.append(dict(estimator=est, method=f"QUBO-{pname}", problem="GMVP", gamma=np.nan,
                            time=s.sample_time, feasible_rate=s.feasible_rate, repaired=s.repaired,
                            vol_ratio=np.sqrt(w @ C @ w / var_star),
                            round_vol_ratio=np.sqrt(wr @ C @ wr / var_star),
                            sharpe=mvo.sharpe(w.values, m, C), vol=np.sqrt(w @ C @ w)))

            # QUBO：均值—變異數 gamma 掃描 → 取事前 Sharpe 最高者為 MSRP
            cands = {"GMVP": w}
            for g in gammas:
                w, s = done[(pname, est, "MV", g)]
                cands[g] = w
                f_q = mvo.mv_objective(w.values, m, C, g)
                f_c = mvo.mv_objective(cont_mv[g].values, m, C, g)
                wr = mvo.round_to_lots(cont_mv[g], P).values
                log.append(dict(estimator=est, method=f"QUBO-{pname}", problem="MV", gamma=g,
                                time=s.sample_time, feasible_rate=s.feasible_rate, repaired=s.repaired,
                                ce_loss=f_q - f_c,                      # 確定等值報酬損失（季）
                                round_ce_loss=mvo.mv_objective(wr, m, C, g) - f_c,
                                sharpe=mvo.sharpe(w.values, m, C), vol=np.sqrt(w @ C @ w)))
            best = max(cands, key=lambda k: mvo.sharpe(cands[k].values, m, C))
            weights[strategy_name(f"QUBO-{pname}", est, "MSRP")] = cands[best]
            log.append(dict(estimator=est, method=f"QUBO-{pname}", problem="MSRP-select",
                            gamma=np.nan if best == "GMVP" else best,
                            sharpe=mvo.sharpe(cands[best].values, m, C)))

        for port, w in (("GMVP", w_g), ("MSRP", w_s)):
            log.append(dict(estimator=est, method="MVO", problem=port,
                            sharpe=mvo.sharpe(w.values, m, C), vol=np.sqrt(w @ C @ w)))

    weights.update(convex_regularised(cfg, mu, covs))
    weights["EqualWeight"] = pd.Series(1 / len(mu), index=mu.index)
    W = pd.DataFrame(weights).fillna(0.0)
    W.index.name = "stock_id"
    L = pd.DataFrame(log)
    L.insert(0, "quarter", win.quarter)
    return W, L


def run_quarter(cfg: dict, win: QuarterWindow, prices: pd.DataFrame, force_forecast: bool = False) -> None:
    tickers, stocks = universe_for(cfg, win)
    mu = step_forecast(cfg, win, stocks, force=force_forecast)
    tickers = [t for t in tickers if t in mu.index]
    mu = mu.loc[tickers]
    covs = step_covariances(cfg, win, prices, tickers)
    W, L = step_solve(cfg, win, mu, covs)

    # 事後 oracle（只作診斷）
    o_mu, o_cov = oracle_inputs(prices, win, tickers, cfg["covariance"]["frequency"])
    W["Oracle|Realized|GMVP"] = mvo.gmvp(o_mu, o_cov).reindex(W.index).fillna(0)
    W["Oracle|Realized|MSRP"] = mvo.max_sharpe(o_mu, o_cov).reindex(W.index).fillna(0)

    out = quarter_dir(cfg, win.quarter)
    W.to_csv(out / "weights.csv")
    L.to_csv(out / "solver_log.csv", index=False)
    o_mu.rename_axis("stock_id").to_csv(out / "realized_return.csv")
    meta = win.as_dict() | {"n_assets": len(tickers), "tickers": tickers,
                            "cond_number": {e: covariance.condition_number(c) for e, c in covs.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prices(cfg: dict) -> pd.DataFrame:
    ids = sorted(p.stem for p in Path(cfg["paths"]["stocks"]).glob("*.feather"))
    return close_panel(load_stocks(cfg["paths"]["stocks"], ids))
