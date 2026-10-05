"""靜態圖表（matplotlib，GitHub 上可直接預覽）。圖中文字使用英文以避免字型問題。"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

METHOD_COLORS = {
    "MVO": "#1f77b4",
    "MVO-round-P20": "#9ecae1",
    "MVO-round-P100": "#6baed6",
    "SA-cont": "#ff7f0e",
    "QUBO-P20": "#2ca02c",
    "QUBO-P100": "#d62728",
    "MVO-cap10": "#9467bd",
    "EqualWeight": "#7f7f7f",
    "0050 ETF": "#000000",
}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def nav_chart(daily: pd.DataFrame, estimator_label: str, schedule, ax=None, title=""):
    methods = ["MVO", "SA-cont", "QUBO-P20", "QUBO-P100", "MVO-cap10"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True) if ax is None else (None, ax)
    for a, port in zip(axes, ["GMVP", "MSRP"]):
        for m in methods:
            col = f"{m}|{estimator_label}|{port}"
            if col in daily:
                # MVO 與連續 SA 幾乎重合：MVO 畫成較粗的半透明底線
                style = dict(lw=4, alpha=0.35) if m == "MVO" else dict(lw=1.6)
                label = "MVO (≈ SA-cont)" if m == "MVO" else m
                a.plot((1 + daily[col]).cumprod() - 1, label=label, color=METHOD_COLORS[m], **style)
        for b in ["EqualWeight", "0050 ETF"]:
            a.plot((1 + daily[b]).cumprod() - 1, label=b, color=METHOD_COLORS[b], lw=1.2, ls="--")
        for w in schedule[1:]:
            a.axvline(w.announcement, color="grey", lw=0.6, ls=":")
        a.set_title(f"{port} ({estimator_label} covariance)")
        a.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Cumulative return (after costs)")
    axes[1].legend(loc="upper left", fontsize=8)
    if fig is not None:
        fig.suptitle(title or "Walk-forward out-of-sample performance", fontweight="bold")
        fig.autofmt_xdate()
        fig.tight_layout()
    return fig


def solver_quality_chart(quality_mv: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    floor = 0.01  # bp；四捨五入基準的損失可能為 0 或數值誤差造成的極小負值
    for m in ["QUBO-P20", "QUBO-P100"]:
        d = quality_mv.loc[m]
        p = m.split("-")[1]
        ax.plot(d.index, (d["ce_loss"] * 1e4).clip(lower=floor), marker="o",
                color=METHOD_COLORS[m], label=f"QUBO {p} + simulated annealing")
        ax.plot(d.index, (d["round_ce_loss"] * 1e4).clip(lower=floor), marker="x", ls="--",
                color=METHOD_COLORS[f"MVO-round-{p}"], label=f"Convex optimum rounded to {p} lots")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Risk aversion $\gamma$ (quarterly units)")
    ax.set_ylabel("Certainty-equivalent loss vs. continuous\noptimum (basis points / quarter, log)")
    ax.set_title("Solution quality on the identical mean-variance objective", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def weights_chart(W: pd.DataFrame, estimator_label: str, names: dict[str, str] | None = None, top: int = 20):
    methods = ["MVO", "SA-cont", "QUBO-P20", "QUBO-P100"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    for a, port in zip(axes, ["GMVP", "MSRP"]):
        cols = [f"{m}|{estimator_label}|{port}" for m in methods]
        sub = W[cols]
        order = sub.max(axis=1).sort_values(ascending=False).index[:top]
        x = np.arange(len(order))
        bw = 0.8 / len(methods)
        for i, (m, c) in enumerate(zip(methods, cols)):
            a.bar(x + (i - 1.5) * bw, sub.loc[order, c], bw, label=m, color=METHOD_COLORS[m])
        a.set_xticks(x)
        a.set_xticklabels(order, rotation=0, fontsize=8)
        a.yaxis.set_major_formatter(PercentFormatter(1.0))
        a.set_title(f"{port} weights (top {top} names, {estimator_label})")
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def concentration_chart(conc_raw: pd.DataFrame):
    methods = ["MVO", "MVO-round-P20", "MVO-round-P100", "SA-cont", "QUBO-P20", "QUBO-P100", "MVO-cap10"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for a, port in zip(axes, ["GMVP", "MSRP"]):
        d = conc_raw[conc_raw["portfolio"] == port]
        data = [d[d["method"] == m]["eff_n"].values for m in methods]
        bp = a.boxplot(data, patch_artist=True, widths=0.6)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.7)
        a.set_xticks(range(1, len(methods) + 1))
        a.set_xticklabels(methods, rotation=25, fontsize=8)
        a.set_title(f"{port}: effective number of holdings (1/Σw²)")
    fig.tight_layout()
    return fig


def forecast_scatter(preds: pd.DataFrame):
    d = preds.dropna(subset=["actual"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(d["actual"], d["predicted"], s=4, alpha=0.25)
    lim = [min(d["actual"].min(), d["predicted"].min()), max(d["actual"].max(), d["predicted"].max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Realised 63-day return")
    ax.set_ylabel("LSTM predicted 63-day return")
    ax.set_title("Out-of-sample forecasts (all quarters)", fontweight="bold")
    fig.tight_layout()
    return fig


def quarterly_heatmap(quarterly: pd.DataFrame, estimator_label: str):
    rows = []
    for m in ["MVO", "SA-cont", "QUBO-P20", "QUBO-P100", "MVO-cap10"]:
        for port in ["GMVP", "MSRP"]:
            rows.append(f"{m}|{estimator_label}|{port}")
    rows += ["EqualWeight", "0050 ETF"]
    pv = quarterly.pivot_table(index="strategy", columns="quarter", values="return").reindex(rows)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    vmax = np.nanmax(np.abs(pv.values))
    im = ax.imshow(pv.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pv.shape[1]))
    ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(pv.shape[0]))
    ax.set_yticklabels([r.replace(f"|{estimator_label}|", " ") for r in pv.index], fontsize=8)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.1%}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, format=PercentFormatter(1.0))
    ax.set_title(f"Quarterly holding-period returns ({estimator_label})", fontweight="bold")
    fig.tight_layout()
    return fig
