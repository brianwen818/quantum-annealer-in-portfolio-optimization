# 文獻蒐集紀錄（Literature Search Protocol）

本資料夾保存 [`../literature-review.md`](../literature-review.md) 所依據的系統性文獻蒐集結果。蒐集流程使用 [Literature-Review-Skill](https://github.com/brianwen818/Literature-Review-Skill)（期刊檢索 → 評分 → 逐篇閱讀 → 引文回溯 → 篩選 → 匯出），資料來源為 [Crossref](https://www.crossref.org/)。

> **說明：** 本次執行的所有人工審查決定（確認關鍵詞與期刊、接受種子文獻比對、逐篇閱讀摘要並給 1–5 分、在短名單勾選 yes／no）皆由 AI 助理（Claude，於 Claude Code 中執行）代作者完成。相關度分數僅用於排序與篩選，不代表對文獻品質的評價。

## 檔案

| 檔案 | 內容 |
| :-- | :-- |
| `summary.md` | 數量漏斗、各期刊結果、主題分布、最相關文章與理由，以及檢索缺口分析 |
| `all-relevant-articles.xlsx` | 通過相關度門檻的 1,409 篇候選文章（依期刊、再依相關度排序） |
| `shortlist.xlsx` | 短名單 40 篇（`shortlist` 分頁）與引文回溯 22 篇（`snowball` 分頁），含 `add-zotero-or-not` 勾選欄 |
| `zotero-added.xlsx` | 勾選 yes 的 43 篇 |
| `zotero-import.ris` | 上述 43 篇的 RIS 檔，可直接匯入 Zotero、EndNote 等文獻管理軟體 |

## 檢索設定

- **執行日期：** 2026-10-05
- **主要文件（用於計算用詞與語意相似度）：** 研究簡報、量子退火教學簡報，以及作者的文獻筆記（移除內嵌圖片後的純文字版本）
- **種子文獻：** 32 筆（含 Lozano, 2026a/b；Stopfer & Wagner, 2025；Mugel et al., 2022；Rosenberg et al., 2016；Venturelli & Kondratyev, 2019；Markowitz, 1952；Ledoit & Wolf, 2003/2004；Lucas, 2014；Kadowaki & Nishimori, 1998 等）。經 Crossref 比對（標題、第一作者、年份皆須吻合），27 筆解析出 DOI（26 筆自動、1 筆人工確認：Markowitz, 1952）；其餘 5 筆為 arXiv 預印本或會議論文，Crossref 無對應紀錄，另於文獻回顧中以 arXiv 補充。
- **出版年份：** 2005–2026
- **短名單：** n = 40，`global` 模式（全部期刊中最相關的 40 篇），agent 評分至少 3 分
- **期刊（22 本）：**
  - 財務：Quantitative Finance、Journal of Portfolio Management、Journal of Banking & Finance、Computational Economics、Journal of Financial Data Science
  - 作業研究與計算：European Journal of Operational Research、Annals of Operations Research、Computers & Operations Research、INFORMS Journal on Computing、Expert Systems with Applications
  - 量子計算與物理：Quantum Information Processing、IEEE Transactions on Quantum Engineering、npj Quantum Information、Physical Review Applied、Physical Review A、Physical Review Research、Quantum、Quantum Machine Intelligence、EPJ Quantum Technology、Frontiers in Physics
  - 綜合：Scientific Reports、Entropy
- **關鍵詞（30 個，英文）：** quantum annealing portfolio optimization; QUBO portfolio; quadratic unconstrained binary optimization; Ising formulation; D-Wave quantum annealer; hybrid quantum-classical solver; constrained quadratic model; quantum-inspired tensor network; dynamic portfolio optimization; quantum approximate optimization algorithm portfolio; variational quantum eigensolver portfolio; cardinality constrained portfolio; mixed-integer quadratic programming portfolio; penalty parameter QUBO; simulated annealing portfolio; tabu search QUBO; digital annealer; approximation ratio quantum optimization; minor embedding chain break; reverse quantum annealing; quantum computing finance; covariance matrix shrinkage; estimation error mean-variance portfolio; integer portfolio optimization; portfolio weight constraints regularization; benchmarking quantum optimization; optimal trading trajectory; adiabatic quantum computation; quantum advantage combinatorial optimization; LSTM stock return prediction
  - 自動抽取的 30 個詞中保留 7 個（部分改寫為較完整的片語），其餘因過於籠統（如 *portfolio optimization*）、屬筆記用語（如 *gmvp msrp*、*apa citation*）或離題（如 *government bond*）而刪除；另由 AI 助理補入 23 個領域標準用語。由於主要文件以中文為主，自動抽取的英文詞彙有限，這一步的人工判斷比例較高。

## 評分方式

每篇候選文章以三個訊號的百分位加權平均排序，前 20%（依有無摘要分層）列為「過門檻」：

| 訊號 | 意義 | 權重 |
| :-- | :-- | :-- |
| 引文重疊（coupling） | 引用種子文獻的篇數（直接引用加權 3 倍），以及與種子文獻參考文獻的重疊 | 0.40 |
| 語意相似（SPECTER） | 以 `allenai-specter` 模型比較標題與摘要和主要文件的語意相似度（RTX 4070 GPU） | 0.40 |
| 用詞相似（TF-IDF） | 標題與摘要和主要文件的用詞相似度 | 0.20 |

排名最前的 120 篇（n × 3）由 AI 助理逐篇閱讀標題與摘要，依 5 = 核心、4 = 直接有用、3 = 背景、2 = 弱相關、1 = 離題給分並寫下理由；最終排序以閱讀評分為主。

## 各階段數量

| 階段 | 數量 |
| :-- | --: |
| 查詢次數（30 關鍵詞 × 22 期刊） | 660 |
| 檢索命中 | 11,255 |
| 不重複文章 | 7,132 |
| 排除（書評、勘誤、目錄等 44 篇；重複標題 18 篇；種子本身 6 篇） | 68 |
| 進入評分 | 7,064 |
| 過門檻（all-relevant-articles.xlsx） | 1,409 |
| 逐篇閱讀（期刊檢索） | 120（5 分 4 篇、4 分 8 篇、3 分 33 篇、2 分 36 篇、1 分 39 篇） |
| 短名單 | 40（勾選 yes 26 篇） |
| 引文回溯候選（被 ≥ 2 個來源引用） | 250 |
| 逐篇閱讀（引文回溯） | 60（5 分 2 篇、4 分 5 篇、3 分 15 篇、2 分 19 篇、1 分 19 篇） |
| 引文回溯列出 | 22（勾選 yes 17 篇） |
| 匯出 RIS | 43 |

**勾選原則：** 相關度 4–5 分一律 yes；3 分者，僅在直接支持本研究論點（離散化、懲罰項、公平比較、估計誤差與正則化、基數限制的古典基線）時勾選 yes，偏向純物理理論或與投組無關的應用則為 no。

## 補充檢索與限制

- 本領域的關鍵文獻多以 arXiv 預印本發表（例如 Lozano, 2026a/b；Stopfer & Wagner, 2025；Hodson et al., 2019；Palmer et al., 2021, 2022），Crossref 期刊檢索無法涵蓋。這些文獻另以 arXiv API、Semantic Scholar 與網路檢索補充，並逐筆核對題名、作者、年份與 arXiv 編號後才納入文獻回顧。
- D-Wave CQM 求解器支援的變數型態等技術細節，以 D-Wave 官方文件為準。
- Crossref 約 33% 的候選文章附有摘要；無摘要者僅能依標題與期刊判斷。
- Crossref 幾乎不收錄中文文獻，本次未涵蓋中文與台灣本地的研究。
- 評分與 AI 助理的閱讀判斷並非完全可重現；Crossref 資料亦持續更新，重跑時數量可能略有差異。
