# 資料說明（Data Card）

本研究使用的資料皆已收錄於此資料夾，clone 後即可重現所有結果，不需另外下載。
資料由 NCCU AI QC Lab 同意公開，僅供學術研究與結果重現使用。

| 檔案 | 內容 | 來源 | 期間 |
|---|---|---|---|
| `stocks/<代碼>.feather` | 61 檔個股的**還原**日線（2025/03–2026/03 各季 0050 成分股的聯集） | [FinMind](https://finmindtrade.com/) `TaiwanStockPriceAdj` | 各股上市日（最早 1994-10）～ 2026-07-16 |
| `meta/Holding_0050.xlsx` | 元大台灣 50（0050）每月持股明細（代碼、名稱、權重…） | TEJ 台灣經濟新報 | 2003/06 ～ 2026/06 |
| `meta/trading_dates.feather` | 台股交易日曆 | FinMind | 1999 ～ 2026 |
| `meta/0050_benchmark.feather` | 0050 ETF 還原日線（作為大盤基準） | Yahoo Finance `0050.TW`（`auto_adjust=True`） | 2003-06 ～ 2026-07-16 |

## 欄位

`stocks/*.feather`（FinMind 原始欄位）：

| 欄位 | 說明 |
|---|---|
| `date` | 交易日（字串 `YYYY-MM-DD`） |
| `stock_id` | 股票代碼 |
| `Trading_Volume` / `Trading_money` / `Trading_turnover` | 成交股數／成交金額／成交筆數 |
| `open` / `max` / `min` / `close` | 還原開高低收 |
| `spread` | 漲跌價差 |

`meta/Holding_0050.xlsx`：`年月`（例 `2025/03`）、`標的碼`、`標的名稱`、`投資比率％` 等；
`標的碼` 為非數字者（如 `M2300`、`MTSE`）是產業／指數彙總列，程式會自動排除。

## 備註

* 0050 ETF 於 2025 年 6 月進行 1 拆 4 分割，Yahoo Finance 的還原價已處理；
  2025-06-11 ～ 06-17 分割停牌期間價格為常數。
* 原始資料夾中的 `0050.feather` 內容誤為 2325（矽品）的資料，已改以 Yahoo Finance 重新取得，並未使用該錯誤檔案。
* `scripts/download_data.py` 可重新下載（FinMind 還原股價需付費等級的 API token）。
