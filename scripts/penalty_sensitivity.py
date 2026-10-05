"""QUBO 超參數敏感度：penalty 係數 × sweeps 對解品質的影響。

只使用第一季決策日的「事前」輸入（μ 與主估計式 Σ），以目標函數評估，不使用任何未來資料，
因此可用來選定 walk-forward 實驗的 penalty_factor 而不造成 look-ahead。

輸出：results/summary/penalty_sensitivity.csv、results/figures/penalty_sensitivity.png
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from qaportfolio import mvo, qubo  # noqa: E402
from qaportfolio.config import load_config  # noqa: E402

FACTORS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
SWEEPS = [2000, 8000, 32000]
GAMMAS = [None, 5.0]          # GMVP 與一個代表性的均值—變異數問題


def one(mu, cov, P, gamma, factor, sweeps, reads, seed):
    s = qubo.solve(mu, cov, P, gamma, "neal", {"num_reads": reads, "num_sweeps": sweeps}, factor, seed)
    return s


def main() -> None:
    cfg = load_config()
    q = cfg["quarters"][0]
    d = cfg["paths"]["results"] / q
    mu = pd.read_csv(d / "mu.csv", dtype={"stock_id": str}).set_index("stock_id")["mu"]
    cov = pd.read_csv(d / f"cov_{cfg['covariance']['primary']}.csv", index_col=0)
    cov.index = cov.index.astype(str)
    cov.columns = cov.columns.astype(str)
    cov = cov.loc[mu.index, mu.index]
    m, C = mu.values, cov.values

    grid = list(itertools.product([20, 100], GAMMAS, FACTORS, SWEEPS))
    sols = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(
        delayed(one)(mu, cov, P, g, f, s, 100, cfg["seed"]) for P, g, f, s in grid)

    rows = []
    w_g = mvo.gmvp(mu, cov).values
    for (P, g, f, s), sol in zip(grid, sols):
        w = sol.weights
        if g is None:
            loss = np.sqrt(w @ C @ w / (w_g @ C @ w_g)) - 1           # 相對波動度超額
        else:
            w_c = mvo.max_utility(mu, cov, g).values
            loss = mvo.mv_objective(w, m, C, g) - mvo.mv_objective(w_c, m, C, g)
        rows.append(dict(precision=P, problem="GMVP" if g is None else f"MV(gamma={g:g})",
                         penalty_factor=f, num_sweeps=s, loss=loss, feasible_rate=sol.feasible_rate,
                         repaired=sol.repaired, time=sol.sample_time))
    df = pd.DataFrame(rows)
    out = cfg["paths"]["results"] / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "penalty_sensitivity.csv", index=False, float_format="%.6g")
    print(df.pivot_table(index=["precision", "problem", "penalty_factor"], columns="num_sweeps",
                         values=["loss", "feasible_rate"]).round(4).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, ((P, prob), g) in zip(axes, df.groupby(["precision", "problem"])):
        for s, gg in g.groupby("num_sweeps"):
            ax.plot(gg["penalty_factor"], gg["loss"], marker="o", label=f"{s} sweeps")
        ax.set_xscale("log")
        ax.set_title(f"P={P}, {prob}")
        ax.set_xlabel("penalty factor (λ / L)")
    axes[0].set_ylabel("Loss vs. continuous optimum\n(GMVP: excess vol ratio; MV: CE loss / quarter)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    figs = cfg["paths"]["results"] / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "penalty_sensitivity.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
