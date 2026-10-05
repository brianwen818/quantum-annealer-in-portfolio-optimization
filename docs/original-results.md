# 原始版本（2026 春季期末報告）與重構版本的差異

本研究最初以單一 Jupyter notebook 完成（封存於 [`archive/ch6_portfolio_optimization_original.ipynb`](../archive/ch6_portfolio_optimization_original.ipynb)），
並於 AI QC Lab 期末發表（[簡報 PDF](slides/Quantum_Annealing_Portfolio_Optimization.pdf)）。
整理成公開 repo 時，我重新檢查了整個流程，發現數個會影響結論的問題，因此重構程式並重新執行實驗。
本文件記錄原始結果、發現的問題與修正方式，讓讀者能對照簡報與 repo 中的數字。

## 1. 原始版本的設定與主要結果

* **期間**：僅 2025/03 一季（2025-03-07 決策，持有至 2025-06-05）。
* **輸入**：LSTM 預測季報酬（樣本外 Rank IC 0.150、方向準確率 55.6%、R² 0.046）；
  6 種共變異數（含一個以未來資料計算的 oracle）。
* **Solver**：PyPortfolioOpt（MVO）、`scipy.dual_annealing`（標為 SA）、
  dimod CQM → BQM + `neal`（標為「QA20／QA100」，精度 5% 與 1%）。
* **簡報中的主要結論**：
  * GMVP：MVO 與 oracle 權重的誤差最小（EWMA 1.66%），QA 皆大於 3%。
  * MSRP：QA100 與 oracle 權重的誤差最小（Sample 3.52%）；以累積報酬、Calmar、MDD 計算勝場，QA100 共 17 勝、QA20 9 勝、SA 6 勝、MVO 4 勝。
  * 推論：「QA 的離散性帶來被動正則化」、「精度 100 反而降低搜尋品質」、「50 檔資產仍屬凸問題，量子優勢應在 NP-hard 情境」。

## 2. 發現的問題與修正

| # | 問題 | 影響 | 重構後的作法 |
|---|---|---|---|
| 1 | **未使用量子硬體**：標為「D-Wave Quantum Annealing／QA」的方法實際是 `neal.SimulatedAnnealingSampler`（古典模擬退火） | 命名誤導 | 一律改稱「QUBO + 模擬退火（neal）」；新增 `samplers.py`，設定 `DWAVE_API_TOKEN` 後可切換到 QPU 或 Leap hybrid，但本 repo 所有結果皆來自 `neal` |
| 2 | **三種方法的「MSRP」不是同一個問題**：MVO 最大化 Sharpe；SA 在連續空間最大化 Sharpe；QUBO 最小化 $0.5\,x^\top\Sigma x-\mu^\top x$（$x$ 為 lot 數），換算成權重後等於風險趨避係數 $\gamma = 0.5P$，QA20 與 QA100 因此在解兩個不同的均值—變異數問題 | 「QA100 在 MSRP 勝出」無法歸因於 solver | QUBO 版 MSRP 改為：在 $\gamma$ 網格上求解 $\tfrac{\gamma}{2}w^\top\Sigma w-\mu^\top w$，取事前 Sharpe 最高者。另以**完全相同的目標函數**比較 QUBO 解與連續最佳解，量化解的品質 |
| 3 | 只有一季、一次回測 | 樣本數 = 1，勝場統計不具意義 | 延伸為 2025/03–2026/03 共 5 季 walk-forward，每季以當時資料重新訓練 LSTM |
| 4 | oracle（以未來資料計算的「正解」）與一般策略一起排名，並以「與 oracle 權重的距離」作為準確度 | 混淆診斷與策略績效 | oracle 只作為診斷指標，不列入任何排名 |
| 5 | `0050.feather` 實際是 2325（矽品）的資料 | 「vs 0050」比較錯誤 | 以 Yahoo Finance 的 0050 ETF 還原價取代 |
| 6 | 回測漏掉決策日到下一交易日的報酬；季與季之間有空窗；權重每日再平衡；未計交易成本 | 報酬略有偏差 | 公告日收盤建倉、季內買進持有、下一公告日收盤再平衡；計入手續費 0.1425% 與證交稅 0.3% |
| 7 | LSTM 沒有固定亂數種子；notebook 有亂序執行與過期輸出（說明文字中的 IC 0.148 與實際輸出 0.150 不符） | 無法重現 | 全流程固定 seed、TensorFlow deterministic ops，以腳本一鍵重跑 |
| 8 | `stock_id` 被當作 LSTM 特徵，經各股 scaler 標準化後恆為 0 | 無資訊的特徵 | 移除 |
| 9 | 共變異數以 0 填補缺值；SA 權重截斷後未重新正規化；QUBO 找不到可行解時默默回傳全 0 權重 | 小幅偏誤／潛在錯誤 | 要求一年以上歷史資料；截斷後重新正規化；QUBO 回報可行率，無可行解時以最大餘數法修復並記錄 |
| 10 | QUBO penalty λ 與 sweeps 為手調（λ = 5 / 15） | 難以解釋 | 以「單位違反可換得的最大目標改善量」L 為尺度，並以只用事前輸入的敏感度實驗（`scripts/penalty_sensitivity.py`）選定 λ = 0.1L；另新增「凸最佳解四捨五入」與「MVO + 10% 權重上限」兩個對照組 |

## 3. 哪些結論仍然成立？

重構後的完整結果見 [README](../README.md#主要結果) 與 `results/summary/`。與原始結論的對照也整理在 README 的「與原始結論的對照」一節。
