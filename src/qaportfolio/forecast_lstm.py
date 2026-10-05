"""以單層 LSTM 預測各股未來一季報酬（作為均值—變異數模型的 mu）。

防止資訊洩漏的設計：
1. 訓練資料只取「標籤在決策日當天或之前已實現」的樣本（t + horizon <= 決策日）。
2. 每檔股票依時間切成訓練／驗證集，兩者之間留 horizon 天的 purge，避免標籤重疊。
3. StandardScaler 只在訓練段 fit。
4. 預測時只使用決策日（含）以前的特徵視窗。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, TARGET, add_features


@dataclass
class ForecastResult:
    mu: pd.Series                 # 決策日對各股的預測季報酬
    predictions: pd.DataFrame     # 樣本外預測：date, stock_id, predicted, actual
    history: pd.DataFrame         # 訓練過程 loss
    n_train: int
    n_val: int


def _windows(x: np.ndarray, window: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(x, window, axis=0).transpose(0, 2, 1)


def _build_model(n_features: int, cfg: dict):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU

    model = Sequential([
        Input(shape=(cfg["window"], n_features)),
        LSTM(cfg["lstm_units"], activation="tanh"),
        Dropout(cfg["dropout"]),
        Dense(cfg["dense_units"]),
        LeakyReLU(negative_slope=0.1),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_forecast(stocks: dict[str, pd.DataFrame], decision_date: pd.Timestamp,
                       cfg: dict, seed: int, verbose: int = 0) -> ForecastResult:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    window, horizon = cfg["window"], cfg["horizon"]
    X_tr, y_tr, X_va, y_va = [], [], [], []
    scalers, pred_inputs = {}, {}

    for sid, raw in stocks.items():
        full = add_features(raw, horizon)
        hist = full[full["date"] <= decision_date].dropna(subset=FEATURES).reset_index(drop=True)
        if len(hist) < 2 * window + horizon:
            continue

        # 決策日當下標籤已實現的樣本：最後 horizon 列的標籤要到決策日之後才知道
        labelled = hist.iloc[: len(hist) - horizon]
        split = int(len(labelled) * (1 - cfg["val_frac"]))
        train_part = labelled.iloc[:split]
        val_part = labelled.iloc[split + horizon:]

        scaler = StandardScaler().fit(train_part[FEATURES].values)
        scalers[sid] = scaler

        def seq(part):
            x = scaler.transform(part[FEATURES].values).astype(np.float32)
            return _windows(x, window), part[TARGET].values[window - 1:].astype(np.float32)

        Xs, ys = seq(train_part)
        X_tr.append(Xs); y_tr.append(ys)
        if len(val_part) >= window:
            Xs, ys = seq(val_part)
            X_va.append(Xs); y_va.append(ys)

        # 樣本外預測：決策日前 horizon 個交易日（其標籤皆在決策日之後才實現）
        x_all = scaler.transform(hist[FEATURES].values).astype(np.float32)
        tail_idx = np.arange(len(hist) - horizon, len(hist))
        actual = full.set_index("date")[TARGET]
        pred_inputs[sid] = (
            np.stack([x_all[i - window + 1: i + 1] for i in tail_idx]),
            hist["date"].values[tail_idx],
            actual.reindex(hist["date"].values[tail_idx]).values,
        )

    X_tr, y_tr = np.concatenate(X_tr), np.concatenate(y_tr)
    X_va, y_va = np.concatenate(X_va), np.concatenate(y_va)

    model = _build_model(len(FEATURES), cfg)
    hist_cb = model.fit(
        X_tr, y_tr, validation_data=(X_va, y_va),
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], shuffle=True,
        callbacks=[EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                                 restore_best_weights=True)],
        verbose=verbose,
    )

    rows = []
    for sid, (X, dates, actual) in pred_inputs.items():
        p = model.predict(X, verbose=0).ravel()
        rows.append(pd.DataFrame({"date": dates, "stock_id": sid,
                                  "predicted": p, "actual": actual}))
    preds = pd.concat(rows, ignore_index=True)
    mu = preds[preds["date"] == decision_date].set_index("stock_id")["predicted"].rename("mu")

    return ForecastResult(mu=mu, predictions=preds, history=pd.DataFrame(hist_cb.history),
                          n_train=len(X_tr), n_val=len(X_va))


def forecast_metrics(preds: pd.DataFrame) -> dict:
    """樣本外預測品質：MAE、RMSE、R²、平均每日 Rank IC、方向準確率。"""
    d = preds.dropna(subset=["actual", "predicted"])
    err = d["predicted"] - d["actual"]
    ss_res = (err ** 2).sum()
    ss_tot = ((d["actual"] - d["actual"].mean()) ** 2).sum()
    ic = d.groupby("date")[["actual", "predicted"]].apply(
        lambda g: g["actual"].corr(g["predicted"], method="spearman")).dropna()
    return {
        "n": len(d),
        "MAE": err.abs().mean(),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": 1 - ss_res / ss_tot,
        # 相鄰日期的季報酬標籤高度重疊，故不報告 IC 的 t 統計量
        "RankIC": ic.mean(),
        "DirAcc": ((d["predicted"] > 0) == (d["actual"] > 0)).mean(),
    }
