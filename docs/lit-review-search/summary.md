# 文獻蒐集總結

## 本次設定

- **短名單篇數 (n)**: 40
- **期刊數 (m)**: 22
- **短名單模式**: global
- **出版年份**: 2005 - 2026
- **評分訊號**: coupling, tfidf, specter
- **關鍵詞**: quantum annealing portfolio optimization; QUBO portfolio; quadratic unconstrained binary optimization; Ising formulation; D-Wave quantum annealer; hybrid quantum-classical solver; constrained quadratic model; quantum-inspired tensor network; dynamic portfolio optimization; quantum approximate optimization algorithm portfolio; variational quantum eigensolver portfolio; cardinality constrained portfolio; mixed-integer quadratic programming portfolio; penalty parameter QUBO; simulated annealing portfolio; tabu search QUBO; digital annealer; approximation ratio quantum optimization; minor embedding chain break; reverse quantum annealing; quantum computing finance; covariance matrix shrinkage; estimation error mean-variance portfolio; integer portfolio optimization; portfolio weight constraints regularization; benchmarking quantum optimization; optimal trading trajectory; adiabatic quantum computation; quantum advantage combinatorial optimization; LSTM stock return prediction

## 數量漏斗

| 階段 | 數量 |
|---|---|
| 主要文件 | 3 |
| 種子文獻（解析出 DOI／去重後） | 27 / 32 |
| 查詢次數（關鍵詞 × 期刊） | 660 |
| 檢索命中 | 11255 |
| 不重複文章 | 7132 |
| 排除：書評等非研究論文 | 44 |
| 排除：標題重複 | 18 |
| 排除：本身是種子／已在 Zotero | 6 / 0 |
| 進入評分 | 7064 |
| agent 細讀 | 120 |
| all-relevant-articles.xlsx | 1409 |
| 短名單 | 40 |
| 引文回溯：候選／列出 | 250 / 22 |

## 各期刊結果

| 期刊 | 候選 | 過門檻 | agent 判定相關 | 短名單 |
|---|---|---|---|---|
| Scientific Reports | 463 | 94 | 4 | 3 |
| European Journal of Operational Research | 466 | 83 | 3 | 3 |
| INFORMS Journal on Computing | 307 | 50 | 3 | 3 |
| Journal of Banking & Finance | 236 | 44 | 3 | 3 |
| Physical Review Applied | 353 | 63 | 3 | 3 |
| Quantitative Finance | 253 | 67 | 3 | 3 |
| Quantum Machine Intelligence | 186 | 71 | 3 | 3 |
| Physical Review Research | 432 | 97 | 3 | 2 |
| Annals of Operations Research | 406 | 74 | 2 | 2 |
| Entropy | 403 | 59 | 2 | 2 |
| IEEE Transactions on Quantum Engineering | 199 | 63 | 2 | 2 |
| Quantum | 353 | 81 | 2 | 2 |
| npj Quantum Information | 278 | 54 | 2 | 2 |
| Frontiers in Physics | 313 | 22 | 2 | 1 |
| Quantum Information Processing | 364 | 74 | 2 | 1 |
| Computational Economics | 294 | 44 | 1 | 1 |
| Computers & Operations Research | 385 | 46 | 1 | 1 |
| EPJ Quantum Technology | 176 | 41 | 1 | 1 |
| Journal of Portfolio Management | 195 | 54 | 1 | 1 |
| Physical Review A | 469 | 138 | 1 | 1 |
| Expert Systems with Applications | 431 | 62 | 1 | 0 |
| Journal of Financial Data Science | 102 | 28 | 0 | 0 |

## 主題分布（短名單）

| 主題 | 數量 |
|---|---|
| Estimation error & regularization | 8 |
| Hybrid/decomposition & benchmarks | 7 |
| Classical MIQP & cardinality | 7 |
| Gate-based (QAOA/VQE) portfolio | 7 |
| QUBO modeling & penalties | 5 |
| QA/QUBO portfolio empirics | 4 |
| Annealing physics & control | 2 |

## 最相關的文章

1. **A real-world test of portfolio optimization with quantum annealing** — Sakuler, Wolfgang; Oberreuter, Johannes M.; Aiolfi, Riccardo; Asproni, Luca; Roman, Branislav; Schiefer, Jürgen (2025), *Quantum Machine Intelligence*. 理由: Real-world test of portfolio optimization on a quantum annealer; directly comparable empirical evidence for the solver-vs-investment-performance question.
2. **Backtesting Quantum Computing Algorithms for Portfolio Optimization** — Carrascal, Ginés; Hernamperez, Paula; Botella, Guillermo; Barrio, Alberto del (2024), *IEEE Transactions on Quantum Engineering*. 理由: Backtests quantum portfolio algorithms over time; directly addresses the backtest-design gap the student criticises.
3. **Best practices for portfolio optimization by quantum computing, experimented on real quantum devices** — Buonaiuto, Giuseppe; Gargiulo, Francesco; De Pietro, Giuseppe; Esposito, Massimo; Pota, Marco (2023), *Scientific Reports*. 理由: Best-practice study running portfolio QUBOs on real quantum devices (annealer and gate-based) with guidance on formulation and parameters.
4. **Solving Multiple Discretization Portfolio Optimization Problem with Quantum-Classical Hybrid Algorithms** — Wei, Haijing; Wang, Yanbo J.; Yang, Haoxiang; Yang, Xuan; Cao, Mingming; Xu, Qi; et al. (2025), *Computational Economics*. 理由: Multiple weight-discretization portfolio problems solved with quantum-classical hybrid algorithms; directly about discretization, the study's central mechanism.
5. **Dynamic Asset Allocation with Expected Shortfall via Quantum Annealing** — Xu, Hanjing; Dasgupta, Samudra; Pothen, Alex; Banerjee, Arnab (2023), *Entropy*. 理由: Hybrid quantum-classical annealing for dynamic asset allocation with an expected-shortfall target; extends QUBO portfolio work beyond variance risk.
6. **Decomposition pipeline for large-scale portfolio optimization with applications to near-term quantum computing** — Acharya, Atithi; Yalovetzky, Romina; Minssen, Pierre; Chakrabarti, Shouvanik; Shaydulin, Ruslan; Raymond, Rudy; et al. (2025), *Physical Review Research*. 理由: Decomposition pipeline for large constrained portfolio and rebalancing problems targeting near-term quantum hardware; addresses the scale limit discussed for direct QPU.
7. **A Scalable Algorithm for Sparse Portfolio Selection** — Bertsimas, Dimitris; Cory-Wright, Ryan (2022), *INFORMS Journal on Computing*. 理由: Scalable exact/near-exact algorithm for sparse (cardinality-constrained) portfolio selection; strong classical baseline for NP-hard settings.
8. **When do improved covariance matrix estimators enhance portfolio optimization? An empirical comparative study of nine estimators** — Pantaleo, Ester; Tumminello, Michele; Lillo, Fabrizio; Mantegna, Rosario N. (2011), *Quantitative Finance*. 理由: Empirical comparison of nine covariance estimators in portfolio optimization; supports the study's multi-estimator design.
9. **Quantum annealing for combinatorial optimization: a benchmarking study** — Kim, Seongmin; Ahn, Sang-Woo; Suh, In-Saeng; Dowling, Alexander W.; Lee, Eungkyu; Luo, Tengfei (2025), *npj Quantum Information*. 理由: Benchmarking study of quantum annealing for combinatorial optimization against classical solvers; methodological reference for fair comparison.
10. **An Efficient Global Optimal Method for Cardinality Constrained Portfolio Optimization** — Xu, Wei; Tang, Jie; Yiu, Ka Fai Cedric; Peng, Jian Wen (2024), *INFORMS Journal on Computing*. 理由: Global optimal method for cardinality-constrained mean-variance; classical exact baseline for the NP-hard extension.
11. **Quantum bridge analytics I: a tutorial on formulating and using QUBO models** — Glover, Fred; Kochenberger, Gary; Hennig, Rick; Du, Yu (2022), *Annals of Operations Research*. 理由: Republished Glover et al. QUBO tutorial (Annals of OR version); standard reference for penalty formulation.
12. **The Art of Avoiding Constraints: A Penalty-free Approach to Constrained Combinatorial Optimization with QAOA** — Angara, Prashanti Priya; Lykov, Danylo; Stege, Ulrike; Alexeev, Yuri; Muller, Hausi (2026), *IEEE Transactions on Quantum Engineering*. 理由: Penalty-free constrained optimization with QAOA; parallels Lozano's penalty-free pipeline on the gate model.
13. **Multiclass portfolio optimization via variational quantum Eigensolver with Dicke state ansatz** — Scursulim, J. V. S.; Langeloh, Gabriel M.; Beltran, Victor L.; Brito, Samuraí (2026), *Scientific Reports*. 理由: VQE with Dicke-state ansatz enforcing diversification/cardinality; relevant as a gate-based comparator that builds constraints into the ansatz instead of penalties.
14. **On portfolio optimization: Imposing the right constraints** — Behr, Patrick; Guettler, Andre; Miebs, Felix (2013), *Journal of Banking & Finance*. 理由: Shows which constraints improve out-of-sample mean-variance portfolios; supports the argument that restricting the solution space acts as regularization.
15. **Quantum walk-based portfolio optimisation** — Slate, N.; Matwiejew, E.; Marsh, S.; Wang, J. B. (2021), *Quantum*. 理由: Quantum-walk optimisation for the Hodson et al. rebalancing problem; gate-model alternative to annealing.

## 引文回溯：期刊檢索以外的重要文獻

1. **Defining and detecting quantum speedup** — Rønnow, Troels F.; Wang, Zhihui; Job, Joshua; Boixo, Sergio; Isakov, Sergei V.; Wecker, David; et al. (2014). 被 6 個來源引用. Defines quantum speedup and how to benchmark annealers fairly; central to the fair-comparison critique.
2. **Strategic Portfolio Optimization Using Simulated, Digital, and Quantum Annealing** — Lang, Jonas; Zielinski, Sebastian; Feld, Sebastian (2022). 被 2 個來源引用. Compares simulated, digital and quantum annealing on portfolio QUBOs; closest design to this study's SA vs QA comparison.
3. **Demonstration of a Scaling Advantage for a Quantum Annealer over Simulated Annealing** — Albash, Tameem; Lidar, Daniel A. (2018). 被 3 個來源引用. Demonstrates scaling advantage of QA over SA on crafted instances; key reference in the QA-vs-SA debate.
4. **Portfolio Optimization: Applications in Quantum Computing** — Marzec, Michael (2016). 被 3 個來源引用. Early handbook chapter on portfolio optimization applications in quantum computing (binary encodings of weights).
5. **The unconstrained binary quadratic programming problem: a survey** — Kochenberger, Gary; Hao, Jin-Kao; Glover, Fred; Lewis, Mark; Lü, Zhipeng; Wang, Haibo; et al. (2014). 被 4 個來源引用. Survey of UBQP models and classical solution methods; reference for classical QUBO heuristics.
6. **The Markowitz Optimization Enigma: Is ‘Optimized’ Optimal?** — Michaud, Richard O. (1989). 被 4 個來源引用. Michaud's 'optimized is not optimal' error-maximization argument; core to the decoupling of solver optimality from investment performance.
7. **Generalized Optimal Trading Trajectories: A Financial Quantum Computing Application** — López de Prado, Marcos (2015). 被 3 個來源引用. Lopez de Prado's generalized optimal trading trajectories, precursor of Rosenberg et al.
8. **Quantum Algorithms for Portfolio Optimization** — Kerenidis, Iordanis; Prakash, Anupam; Szilágyi, Dániel (2019). 被 4 個來源引用. Quantum interior-point algorithms for portfolio optimization; theoretical gate-model speedup claim for the continuous problem.
9. **A Quantum Adiabatic Evolution Algorithm Applied to Random Instances of an NP-Complete Problem** — Farhi, Edward; Goldstone, Jeffrey; Gutmann, Sam; Lapan, Joshua; Lundgren, Andrew; Preda, Daniel (2001). 被 4 個來源引用. Foundational adiabatic quantum algorithm paper; theory background for QA.
10. **Theory of Quantum Annealing of an Ising Spin Glass** — Santoro, Giuseppe E.; Martoňák, Roman; Tosatti, Erio; Car, Roberto (2002). 被 5 個來源引用. Theory of QA on Ising spin glass; foundational.

## 資料品質提醒

- 33% 的文章有摘要；沒有摘要的文章在自己的群組內排名，以免被系統性低估。
- 93% 的文章有公開的參考文獻清單；其餘的引文訊號是「缺少」而非 0。
- 919 篇文章至少引用一篇種子文獻。
- 種子文獻解析出 DOI：27 / 32；其中有參考文獻清單的：24。
- 檢核：檢索本身找回了 6 篇種子文獻，其中 100% 排在門檻之上。
- 2 個被引用的 DOI 不在 Crossref，未列入引文回溯。
- Crossref 幾乎不收錄中文文獻；如有需要請另外檢索。

## 分析

**整體觀察。** 這次在 22 本期刊中細讀 120 篇候選，真正直接處理「以量子退火／QUBO 求解投資組合」的期刊論文只有十餘篇，且集中在 2021 年以後的 *Quantum Machine Intelligence*、*Scientific Reports*、*IEEE Transactions on Quantum Engineering*、*Entropy* 與 *Physical Review Research*；財務與作業研究的主流期刊（*Journal of Banking & Finance*、*EJOR*、*Quantitative Finance*）幾乎沒有刊登量子退火的實證研究，命中的多是共變異數估計、穩健最佳化與基數限制等古典文獻。這反映出本領域的主要成果仍以 arXiv 預印本與會議論文發表，而 Crossref 期刊檢索對此覆蓋不足。

**最重要的文章群。** (1) *量子投組實證*：Sakuler et al.（2025）的實盤測試、Carrascal et al.（2024）的量子演算法回測、Buonaiuto et al.（2023）在真實裝置上的最佳實務，以及 Wei et al.（2025）的多重離散化投組問題，是與本研究「解品質 vs. 投資績效」與「離散化」兩條主軸最直接相關的期刊文獻；引文回溯補上了 Lang et al.（2022）對模擬、數位與量子退火的三方比較，設計上最接近本研究。(2) *公平比較方法論*：Rønnow et al.（2014）對 quantum speedup 的定義、Albash & Lidar（2018）與 Denchev et al.（2016）的 QA vs. SA 證據、Kim et al.（2025）的基準測試，為「benchmark 不一致」的批判提供方法論依據。(3) *估計誤差與正則化*：Michaud（1989）、Behr et al.（2013）、Pantaleo et al.（2011）與種子中的 Jagannathan & Ma（2003）共同支撐「求解器次優性可能等同隱式正則化」的論點。(4) *古典基線*：Bertsimas & Cory-Wright（2022）、Xu et al.（2024）、Bonami & Lejeune（2009）等基數／整數限制的精確或近似演算法，是把研究推向 NP-hard 設定時必須面對的對照組。(5) *QUBO 建模與懲罰*：Glover et al. 的 QUBO 教程、Kochenberger et al.（2014）的 UBQP 綜述、Angara et al.（2026）的無懲罰 QAOA，與 Lozano（2026）的無懲罰管線形成呼應。

**未涵蓋的缺口與補救。** (a) 最核心的 Lozano（2026a, 2026b）、Stopfer & Wagner（2025）、Hodson et al.（2019）都只在 arXiv，種子比對亦無法解析，需以 arXiv／Google Scholar 另行檢索（例如 "portfolio optimization D-Wave hybrid"、"QUBO portfolio discretization"、"penalty tuning QUBO"）。(b) 會議論文（ICCS、IEEE QCE、GECCO、EvoCOP）中有不少關於懲罰係數調整與整數編碼比較的研究，不在期刊清單內，建議加入 *Discrete Optimization* 與 *IEEE Access*，或直接以會議名稱檢索。(c) D-Wave 的技術白皮書與 Ocean 文件（CQM 支援的變數型態、hybrid solver 的運作方式）屬灰色文獻，需由官方文件補充。(d) 中文與台灣市場的量子金融研究在 Crossref 幾乎沒有收錄，需另以華藝、國圖博碩士論文系統檢索。
