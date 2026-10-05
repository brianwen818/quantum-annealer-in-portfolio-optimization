# 量子退火於投資組合最佳化之文獻回顧

**Literature Review: Quantum Annealing and QUBO Formulations for Portfolio Optimization**

> 本文為「以 QUBO／退火求解投資組合最佳化：台灣 50 成分股的 Walk-Forward 實證」研究（國立政治大學資管系 AI 量子計算實驗室）之文獻回顧章節。文獻蒐集流程（期刊檢索、評分、引文回溯與篩選紀錄）見 [`docs/lit-review-search/`](lit-review-search/README.md)。

---

## 1. 摘要

將投資組合最佳化寫成二次無約束二元最佳化（Quadratic Unconstrained Binary Optimization, QUBO）問題，再交由量子退火機（quantum annealer）求解，是量子計算在金融領域最早、也最常被拿來示範的應用之一（Orús et al., 2019; Egger et al., 2020; Herman et al., 2023）。本文回顧此一文獻脈絡的三個層次：其一為理論基礎，包括 Markowitz（1952）的均值—變異數框架、QUBO 與 Ising 模型的對應（Lucas, 2014; Glover et al., 2019），以及量子退火的物理原理（Kadowaki & Nishimori, 1998; Albash & Lidar, 2018a）；其二為建模技術，即連續權重的離散化與二進位編碼、以懲罰項（penalty）處理預算與基數限制，以及 D-Wave 的 BQM／DQM／CQM 模型與混合求解器（hybrid solver）；其三為實證文獻，從早期的 D-Wave 2X 小規模驗證（Rosenberg et al., 2016）到近年的大規模基準測試（Stopfer & Wagner, 2025）與混合求解器的運算來源審計（Lozano, 2026b）。

綜合而言，近期文獻的共識相當一致：在標準、靜態的均值—變異數問題上，Gurobi 等商用混合整數規劃求解器在上千檔資產的規模仍能於數秒內證明全域最佳解，量子方法在「解的品質」與「求解速度」上皆未展現優勢；直接上 QPU 的主要瓶頸來自懲罰項造成的稠密耦合，而非硬體本身；混合求解器的優異表現則主要來自其古典管線。然而，這些研究幾乎都以「與最佳解的差距」（approximation ratio、regret）衡量求解器，隱含「目標函數最佳即投資最佳」的假設；回測設計、資料前處理、評估指標與古典對照組亦缺乏一致標準。本研究正是從這個缺口切入：在含雜訊的報酬預測下，把求解器的解品質與樣本外投資績效分開衡量，並檢驗權重離散化是否扮演隱式正則化（implicit regularization）的角色。

---

## 2. 背景

### 2.1 Markowitz 均值—變異數框架與估計誤差

Markowitz（1952）以預期報酬向量 $\boldsymbol{\mu}\in\mathbb{R}^n$ 與共變異數矩陣 $\boldsymbol{\Sigma}\in\mathbb{R}^{n\times n}$ 描述資產，將投資組合選擇寫成權重 $\mathbf{w}$ 的二次規劃。本研究採用的 long-only 均值—變異數效用最大化可寫為

$$
\min_{\mathbf{w}}\; \frac{\gamma}{2}\,\mathbf{w}^\top\boldsymbol{\Sigma}\,\mathbf{w}-\boldsymbol{\mu}^\top\mathbf{w}
\quad\text{s.t.}\quad \mathbf{1}^\top\mathbf{w}=1,\; \mathbf{w}\ge \mathbf{0},
$$

其中 $\gamma$ 為風險趨避係數。全域最小變異投組（GMVP）只需 $\boldsymbol{\Sigma}$；最大夏普比率投組（MSRP）則最大化 $\boldsymbol{\mu}^\top\mathbf{w}/\sqrt{\mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}}$，因為比值形式非二次式，無法直接寫成 QUBO，實務上常以掃描 $\gamma$ 沿效率前緣取事前夏普最高者近似。

此框架在數學上是凸問題，但在統計上極為脆弱。Chopra 與 Ziemba（1993）指出，預期報酬的估計誤差對最適權重的影響約為變異數誤差的十倍以上；DeMiguel、Garlappi 與 Uppal（2009）更發現，在樣本外，多數最佳化策略難以穩定勝過等權重的 $1/N$ 投組。因應之道主要有兩類：一是改善輸入，例如 Ledoit 與 Wolf（2003, 2004a, 2004b）將樣本共變異數向單因子模型、固定相關係數或單位矩陣收縮（shrinkage）；二是限制解空間，Jagannathan 與 Ma（2003）證明，對 GMVP 加上「不得放空」與權重上限等看似「錯誤」的限制，在數學上等同於對共變異數矩陣做收縮，反而能降低樣本外風險。後者對本研究格外重要：權重離散化同樣是一種對解空間的限制，其效果可能與上述正則化機制同源。

### 2.2 QUBO 與 Ising 模型

QUBO 問題的一般形式為

$$
\min_{\mathbf{x}\in\{0,1\}^m}\; \mathbf{x}^\top \mathbf{Q}\,\mathbf{x}
= \sum_i Q_{ii}x_i + \sum_{i<j} (Q_{ij}+Q_{ji})\,x_i x_j ,
$$

因 $x_i^2=x_i$，線性項可併入對角線。以 $x_i=(1-s_i)/2$ 代換即得 Ising 模型的能量函數

$$
E(\mathbf{s})=\sum_i h_i s_i+\sum_{i<j}J_{ij}s_i s_j,\qquad s_i\in\{-1,+1\},
$$

其中 $h_i$ 為局部磁場、$J_{ij}$ 為自旋間耦合。Lucas（2014）系統性地給出了大量 NP 問題（分割、著色、覆蓋、旅行推銷員等）的 Ising 表示；Glover、Kochenberger 與 Du（2019）則從作業研究角度整理了把限制式轉成二次懲罰項的標準手法。由於 QUBO 一般為 NP-hard，任何能有效取樣其低能量狀態的物理裝置都具有潛在的最佳化價值，這正是量子退火的出發點。

### 2.3 模擬退火與量子退火

**模擬退火（simulated annealing, SA）**（Kirkpatrick et al., 1983）以 Metropolis 準則在解空間中隨機遊走：由狀態 $\mathbf{x}$ 移動至能量較高的 $\mathbf{y}$ 的接受機率為 $\exp\{-[f(\mathbf{y})-f(\mathbf{x})]/T\}$。溫度 $T$ 高時近乎無條件接受，負責探索；隨溫度下降逐漸只接受下坡移動，負責利用。D-Wave Ocean 的 `neal` 即為 QUBO 上的模擬退火，其 `num_sweeps`（降溫步數）與 `num_reads`（獨立重複次數）分別對應退火時間與取樣數。

**量子退火（quantum annealing, QA）**（Kadowaki & Nishimori, 1998）則以橫場（transverse field）提供量子漲落。系統的時變哈密頓量為

$$
H(t)=A(t)\,\underbrace{\Bigl(-\sum_i \sigma_i^x\Bigr)}_{H_0}+B(t)\,\underbrace{\Bigl(\sum_i h_i\sigma_i^z+\sum_{i<j}J_{ij}\sigma_i^z\sigma_j^z\Bigr)}_{H_F},
$$

初始時 $A\gg B$，系統處於 $H_0$ 容易製備的基態；隨後 $A$ 遞減、$B$ 遞增，最終哈密頓量 $H_F$ 的基態即為原問題的最佳解。**絕熱定理**保證只要演化夠慢，系統會停留在瞬時基態；所需時間約與最小能隙 $\Delta_{\min}$ 的平方成反比，$\tau \propto 1/\Delta_{\min}^2$（Albash & Lidar, 2018a）。當問題存在大量能量極接近最佳解的次優解時，$\Delta_{\min}$ 趨近於零，所需時間隨之暴增。

兩種退火翻越能量障壁的機制不同：熱漲落擅長翻越「寬而矮」的障壁，量子穿隧（tunneling）則擅長穿過「高而窄」的障壁，其穿透機率隨障壁寬度指數衰減。這也是量子退火被期待在崎嶇能量地形上優於模擬退火的理論理由。不過，真實硬體運作於有限溫度、有雜訊、且非完全絕熱的條件下，實際表現與理想的絕熱演化有相當差距（Hauke et al., 2020）；量子優勢是否存在，必須以與最佳古典演算法的公平比較來判斷，而不能由物理直覺推論。

---

## 3. 投資組合問題的 QUBO 化

### 3.1 權重離散化與整數編碼

QUBO 只接受二元變數，因此連續權重必須先離散化。常見做法是把資金切成 $P$ 份（lot），令整數 $x_i\in\{0,1,\dots,P\}$ 表示第 $i$ 檔資產分到的份數，$w_i=x_i/P$；再把每個整數以 $K$ 個位元 $b_{ik}\in\{0,1\}$ 表示，$x_i=\sum_{k=1}^{K}c_k\,b_{ik}$。編碼方式決定了位元數、係數動態範圍與能量地形的形狀（Tamura et al., 2021）：

| 編碼 | 係數 $c_k$ | 每檔資產位元數 | 優點 | 缺點 |
| :-- | :-- | :-- | :-- | :-- |
| One-hot | 每個位元代表一個整數值，另加 $\sum_k b_{ik}=1$ 限制 | $P+1$ | 結構直觀，任一數值只有一種表示 | 位元數最多，且需額外懲罰項 |
| Unary（thermometer） | $c_k=1$ | $P$ | 係數動態範圍小，對雜訊較不敏感 | 位元數多，同一數值有多種表示（簡併） |
| Binary | $c_k=2^{k-1}$ | $\lceil\log_2(P+1)\rceil$ | 位元數最少 | 最大係數隨 $P$ 指數成長，可表示範圍可能超過 $P$ |
| Bounded-coefficient binary | $1,2,4,\dots,2^{K-2}$，最後一位補足至恰為 $P$ | $\lceil\log_2(P+1)\rceil$ | 位元少且上界恰為 $P$，不產生超出預算的組合 | 最高位係數仍大，單一位元翻轉可造成大幅能量跳動 |

本研究採用最後一種：$P=20$ 時 $c=(1,2,4,8,5)$，$P=100$ 時 $c=(1,2,4,8,16,32,37)$。對 50 檔資產而言，$P=100$ 的問題約有 350 個二元變數。Arai、Oshiyama 與 Nishimori（2023）從理論上分析了以量子退火處理離散化連續變數的有效性；Wei 等人（2026）則把多種離散化方式用於含不可分割資產的投組問題。離散化本身即是一種對解空間的約束：權重只能取 $1/P$ 的整數倍，最小非零部位為 $1/P$，這一點將在第 7 節與隱式正則化的論點連結。

### 3.2 預算與基數限制的懲罰項

在 QUBO 中，限制式需以平方懲罰項併入目標函數。預算限制 $\sum_i x_i=P$ 即為

$$
\min_{\mathbf{b}}\; \frac{\gamma}{2P^2}\,\mathbf{x}^\top\boldsymbol{\Sigma}\,\mathbf{x}-\frac{1}{P}\boldsymbol{\mu}^\top\mathbf{x}
+\lambda\Bigl(\sum_{i}x_i-P\Bigr)^2,\qquad \mathbf{x}=\mathbf{A}\mathbf{b},
$$

其中 $\mathbf{A}=\mathbf{I}_n\otimes\mathbf{c}^\top$ 將位元映回整數份數。基數限制（恰選 $K$ 檔）若以選擇變數 $y_i\in\{0,1\}$ 表示，則為 $\theta(\sum_i y_i-K)^2$。

**懲罰係數 $\lambda$ 的選擇**是此類模型最棘手的超參數。$\lambda$ 太小，最低能量解會違反限制；$\lambda$ 太大，懲罰項主導能量地形，目標函數的差異被壓縮到硬體的精度與雜訊之下，退火器難以在可行解之間分辨優劣（Glover et al., 2019; Verma & Lewis, 2022）。一個保證可行性的充分條件是：令 $L$ 為「違反一單位限制所能換到的最大目標改善量」，則取 $\lambda>L$ 時，任何違反量 $d\neq 0$ 的整數解都有 $\lambda d^2\ge\lambda|d|>L|d|$，最低能量解必然可行。然而本研究的敏感度實驗顯示，在 $\lambda>L$ 的區間，懲罰項主導能量地形，模擬退火的解與最佳解差距最大；將 $\lambda$ 降到約 $0.1L$ 時解品質明顯改善，代價是部分樣本違反預算、需經修復。可行性保證與搜尋品質之間的取捨，正是上述文獻所描述的困境。

### 3.3 Rank-one 懲罰矩陣造成的稠密耦合

將 $\theta(\sum_{i=1}^{N}x_i-K)^2$ 展開可得

$$
\theta\Bigl(\sum_{i=1}^{N}x_i-K\Bigr)^2
=\underbrace{\theta\,\mathbf{x}^\top\mathbf{1}\mathbf{1}^\top\mathbf{x}}_{\text{二次項}}
\;\underbrace{-\,2\theta K\,\mathbf{1}^\top\mathbf{x}}_{\text{線性項}}
\;+\;\underbrace{\theta K^2}_{\text{常數}} .
$$

三個部分的硬體意義截然不同：常數項只平移能量，不影響最佳解；線性項只改變各量子位元的局部偏置 $h_i$，不消耗任何耦合器；真正的問題在二次項。矩陣 $\theta\,\mathbf{1}\mathbf{1}^\top$ 是一個**稠密的 rank-one 矩陣**：它由單一向量的外積生成，因此秩為 1；其每個元素都等於 $\theta$，沒有任何零元素。這個矩陣直接加到共變異數項 $\boldsymbol{\Sigma}$ 上，使得任意兩個變數之間都出現非零耦合，問題的邏輯圖成為完全圖 $K_N$。

此結構在 D-Wave 硬體上的代價很高。QPU 的物理拓撲是稀疏的（Advantage 的 Pegasus 每個量子位元約 15 條耦合，Advantage2 的 Zephyr 約 20 條；Boothby et al., 2020），完全圖必須透過 minor embedding 以多個物理量子位元組成「鏈」（chain）來表示一個邏輯變數；圖越稠密，鏈越長、越容易斷裂，可嵌入的問題規模也越小（在 Advantage 上可嵌入的完全圖約為 177 個邏輯變數）。Lozano（2026a）的實驗顯示，含基數懲罰的模型在 Pegasus 與 Zephyr 上的鏈斷裂比例高達 83%–92%，幾乎無法產出有效解；移除懲罰項、只保留目標函數，再以古典的貪婪可行性投影做後處理後，鏈斷裂率降至 0.04% 以下。作者據此主張，直接 QPU 求解的瓶頸在於懲罰編碼，而非硬體拓撲。同樣的結構也出現在預算限制上：$\lambda(\sum_i x_i-P)^2$ 在位元空間中展開為 $\lambda\,\mathbf{b}^\top\mathbf{a}\mathbf{a}^\top\mathbf{b}$（$\mathbf{a}=\mathbf{A}^\top\mathbf{1}$ 為各位元代表的份數），同樣是稠密的 rank-one 矩陣。更根本的是，即使不考慮懲罰項，金融資產的樣本共變異數矩陣本身也幾乎沒有零元素，因此均值—變異數 QUBO 天生就是稠密問題；本研究 50 檔 $\times$ 7 位元約 350 個變數的設定，已超出直接嵌入的上限，若要上 QPU，須縮減資產數或精度，或改用混合求解器。

### 3.4 D-Wave 的 BQM、DQM、CQM 與混合求解器

D-Wave 的 Ocean SDK 依變數型態與限制處理能力提供三種模型，前兩者可送入 QPU 或混合求解器，CQM 則只能由混合求解器處理（D-Wave Quantum Inc., n.d.）：

| 特性 | BQM（Binary Quadratic Model） | DQM（Discrete Quadratic Model） | CQM（Constrained Quadratic Model） |
| :-- | :-- | :-- | :-- |
| 變數型態 | 二元 $\{0,1\}$ 或自旋 $\{-1,+1\}$ | 離散變數，每個變數從有限個狀態中擇一 | 二元、整數與實數（連續）變數；實數變數的使用有額外限制，詳下文 |
| 限制處理 | 須手動轉為懲罰項 | 「每個變數恰取一種狀態」由模型內建處理，其餘限制仍須轉為懲罰項 | 原生支援線性與二次的等式、不等式限制 |
| 懲罰係數 | 使用者自行設定 | 使用者自行設定 | 由求解器內部處理 |
| 可用求解器 | QPU（需 embedding）、Leap hybrid BQM solver | Leap hybrid DQM solver | Leap hybrid CQM solver |
| 可行性資訊 | 須自行檢查 | 須自行檢查 | 回傳樣本附 `is_feasible` 標記 |
| 投資組合的表達方式 | 權重須離散化並以位元編碼 | 每檔資產的持有份數可直接為離散狀態 | 可直接寫入 $\sum_i w_i=1$、權重上下界與基數限制 |
| 建模難度 | 高 | 中 | 低，接近一般數學規劃 |

關於 CQM 是否支援連續變數，二手資料常有出入（有的稱僅支援二元與整數，有的稱可直接求解連續權重），原因在於兩者描述的是不同時期的版本：Leap hybrid CQM solver 於 2021 年 10 月推出時僅支援二元與整數變數，其後加入了實數（連續）變數；但依官方文件，實數變數不能帶有二次偏置（`maximum_number_of_quadratic_variables_real` 為 0），亦即只能出現在線性項與線性限制中（D-Wave Quantum Inc., n.d.）。因此，均值—變異數目標中的 $\mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}$ 仍無法直接以實數權重寫入 CQM，實務上依然需要整數化權重。

混合求解器的運作方式是以古典的啟發式演算法分解與搜尋，並在過程中把子問題送往 QPU 取樣。Lozano（2026b）追蹤 SDK 回傳的運算時間後發現，在 5 秒的時間預算中，實際使用 QPU 的時間平均僅約 0.034 秒（約 0.68%），而且以純 CPU 的禁忌搜尋（tabu search）在相同時間預算下作為反事實對照，即可解釋大部分的表現。換言之，CQM 在 $N\le 120$ 時能達到 Gurobi 證明的全域最佳解，主要是其古典管線的成功，而非量子取樣帶來的加速。

---

## 4. 實證文獻整理

### 4.1 主要研究一覽

下表整理與本研究最直接相關的實證研究。文獻來源為期刊檢索（Crossref，22 本期刊）、引文回溯，以及另行以 arXiv 補充的預印本；完整篩選紀錄見 [`lit-review-search/`](lit-review-search/README.md)。

| 研究 | 資料與規模 | 問題設定 | 量子／待測求解器 | 對照組 | 評估指標 | 主要發現 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Rosenberg et al. (2016) | 小規模合成實例 | 多期最適交易軌跡，整數持股、交易成本 | D-Wave 2X QPU | 窮舉法 | 找到最佳解的機率 | 小規模實例可找到最佳軌跡，但成功率隨問題規模下降；規模受限於量子位元數與精度 |
| Venturelli & Kondratyev (2019) | 依真實市場統計生成的參數化實例 | 基數限制下的均值—變異數 | D-Wave 2000Q，正向與反向退火 | 遺傳演算法 | time-to-solution | 以貪婪搜尋結果為起點做反向退火，平均比正向退火快 100 倍以上 |
| Phillipson & Bhatia (2021) | Nikkei 225、S&P 500 子集 | 預算、最低報酬限制下的風險最小化 | D-Wave QPU 與 hybrid solver | 商用求解器 | 目標值、求解時間 | 在所測規模內，D-Wave 的解已接近商用求解器 |
| Grant et al. (2021) | 歷史資料產生的投組實例 | 均值—變異數 QUBO | D-Wave 2000Q，各種退火控制（反向退火、暫停等） | 精確解 | 成功機率、鏈斷裂率 | 退火控制參數對成功率影響顯著，可作為硬體調參的基準 |
| Mugel et al. (2021) | 50 檔資產，1 年 | 動態投組，最短持有期限 | D-Wave 2000Q 取樣 + 古典後篩選 | 一般投組 | 與效率前緣的距離 | 量子取樣搭配古典後處理可得到接近效率前緣的交易軌跡 |
| Mugel et al. (2022) | 52 檔資產（政府債、固定與變動收益證券），8 年日資料 | 動態投組，含交易成本 | D-Wave hybrid、IBM VQE、量子啟發張量網路 | Gekko、窮舉法 | 夏普比率、利潤 | D-Wave hybrid 與張量網路能處理最大規模（相當於 1,272 個全連接量子位元） |
| Palmer et al. (2021) | S&P 100、S&P 500 | 投資區間（investment bands）與目標波動度 | D-Wave Advantage hybrid | — | 報酬、風險 | 實務常見限制可寫入 QUBO，並在當前硬體上求解 |
| Lang et al. (2022) | NYSE 股票與 ETF | 古典預選資產後之可變權重 QUBO | 模擬退火、Fujitsu Digital Annealer、D-Wave Advantage | 隨機投組 | 報酬、變異數、分散度 | 三種退火皆能產生分散良好的投組；為少數同時比較 SA、DA、QA 的研究 |
| Xu et al. (2023) | 真實資產資料 | 以 expected shortfall 為風險目標的動態配置 | D-Wave 2000Q、Advantage | 古典最佳解 | 報酬達成率 | 量子退火解可達古典最佳解 80% 以上的報酬，且資產相關性高時表現較佳 |
| Hodson et al. (2019) | 8 檔股票 | 再平衡，離散 lot、非線性交易成本 | Quantum Alternating Operator Ansatz（理想模擬器） | 精確解、標準 QAOA | 與最佳調整後報酬的差距 | 設計保持可行性的 mixer 並與標準 QAOA 比較；找到距最佳調整後報酬 5% 以內的投組 |
| Brandhofer et al. (2022) | 合成實例 | 基數限制 QUBO | 多種 QAOA 變體（含雜訊模擬） | 精確解 | 近似比、成功機率 | 提出區分難易實例的準則；取樣誤差與閘雜訊大幅影響表現 |
| Buonaiuto et al. (2023) | 真實資產資料 | 二進位編碼的限制二次規劃 | VQE（IBM 真實裝置） | 古典精確解 | 解品質 | 整理 VQE 求解投組的超參數最佳實務 |
| Carrascal et al. (2024) | 真實資料，10,000 次回測實驗 | 均值—變異數 | 4 種量子演算法（含 VQE） | 3 種古典演算法 | 回測報酬與風險 | 首個系統性回測比較；量子方法可匹配或略優於古典方法 |
| Sakuler et al. (2025) | 銀行實際上線之投組問題 | 含變異數上限的 QUBO | D-Wave hybrid solver | 古典精確解 | 與全域最佳解的差距 | 結果與古典全域最佳解一致；重點在變異數限制的編碼與參數自動調整 |
| Stopfer & Wagner (2025) | Nasdaq 1,978 檔（2020–2023），250 個實例，3–1,000 檔 | 最小波動度，含限制 | 量子退火、QAOA（硬體與模擬，最多 30 檔） | MIP（Gurobi）、SA、最陡下降、禁忌搜尋、問題專屬啟發式 | 近似比、可行解比例、樣本數 | MIP 在數秒內證明全部實例最佳；專屬啟發式在固定時間內穩定勝過量子方法 |
| Lozano (2026a) | 最多 49 檔（Fama–French 49 產業） | 基數限制的最大夏普 | D-Wave Advantage／Advantage2 直接 QPU | 貪婪演算法、古典參考解 | 鏈斷裂率、regret | 懲罰項造成 83%–92% 鏈斷裂；無懲罰管線降至 0.04% 以下，regret 低於 0.03% |
| Lozano (2026b) | $N=10$–640 合成實例、Fama–French 49 樣本外資料 | 基數限制的均值—變異數—週轉率 | LeapHybridCQM、LeapHybridBQM | Gurobi、純 CPU 禁忌搜尋 | 目標值、QPU 存取時間 | CQM 在 $N\le120$ 全部達到 Gurobi 最佳解，但 QPU 僅佔約 0.68% 運算時間 |
| Wei et al. (2026) | 含不可分割資產的投組 | 多重離散化的投組問題 | D-Wave QPU 與 hybrid 演算法 | 古典求解器 | 報酬、風險 | 混合演算法在報酬與風險上優於純古典與純量子退火 |
| Morapakula et al. (2026) | 印度股市指數 | 季度再平衡：QA 選股 + 古典配權 | D-Wave hybrid | 指數與基金經理人 | 樣本外報酬 | 端到端管線可得到具競爭力的報酬，但投組規模受限於硬體 |

### 4.2 主題綜整

**第一波：概念驗證（2016–2021）。** 早期研究的主要目的是證明「投資組合問題可以寫成 QUBO 並在退火機上跑得出來」。Rosenberg 等人（2016）延續 López de Prado 對最適交易軌跡的設定，以整數持股與多期交易成本構成 QUBO，在 D-Wave 2X 上求解小規模實例；Venturelli 與 Kondratyev（2019）以反向退火搭配貪婪初始解大幅縮短 time-to-solution；Grant 等人（2021）則把投組問題當作基準，系統性評估退火時間、暫停、鏈強度等硬體控制參數。Phillipson 與 Bhatia（2021）、Cohen 等人（2020a, 2020b）以真實指數成分股說明 D-Wave 解可接近商用求解器。這一波研究的共同特徵是：問題規模小（數十檔以內）、以「能否找到最佳解」作為主要指標，且多半沒有完整的樣本外回測。

**第二波：真實限制與動態問題（2020–2023）。** 隨著 Advantage 系統與 hybrid solver 推出，研究開始納入實務限制與多期設定：Mugel 等人（2021, 2022）處理最短持有期限與交易成本下的動態投組，並把量子啟發的張量網路列為對照；Palmer 等人（2021, 2022）將投資區間、目標波動度與指數追蹤的基數限制寫入 QUBO；Fernández-Lorenzo 等人（2021）以混合量子—古典變分方法處理指數追蹤的基數限制；Mattesi 等人（2023）提出同時考量夏普比率與產業分散的 QUBO 形式；Xu 等人（2023）以 expected shortfall 取代變異數；Lang 等人（2022）則同時比較模擬退火、數位退火與量子退火。這一波研究開始報告夏普比率、累積報酬等投資指標，但資料前處理、回測窗口與對照組的設計仍相當分歧（詳見第 6 節）。

**第三波：審計與大規模基準（2024–2026）。** 近期研究的重心從「做得到」轉向「是否真的有優勢」。Stopfer 與 Wagner（2025）建立 250 個實例的大型基準，證明在最小波動度問題上，MIP 求解器能在數秒內解出上千檔資產的全域最佳解，量子方法僅能測試到 30 檔，且解品質不如問題專屬的貪婪啟發式。Lozano（2026a, 2026b）則分別剖析直接 QPU 與 hybrid solver：前者的失敗來自懲罰項造成的稠密耦合，後者的成功主要歸功於古典管線。Acharya 等人（2025）提出把大型投組問題分解為較小子問題的管線，讓量子裝置有機會處理子問題。值得注意的是，Kim 等人（2025）在大型稠密 Hamiltonian 的基準中報告 hybrid 量子退火在精度與求解時間上皆優於最佳古典求解器，與 Stopfer 與 Wagner 的結論方向相反；兩者的問題類型、時間預算與對照組設定不同，正說明了「公平比較」的設計本身就是決定結論的關鍵。

**閘模型（gate-based）的平行發展。** 對於無整數限制的連續問題，Kerenidis 等人（2019）與 Rebentrost 及 Lloyd（2024）提出以量子線性代數加速的演算法，但需要容錯量子電腦；近期可行的則是 QAOA 與 VQE：Hodson 等人（2019）以保持可行性的 mixer 取代懲罰項；Brandhofer 等人（2022）系統比較 QAOA 變體並分析雜訊影響；Barkoutsos 等人（2020）以 CVaR 改善變分最佳化；Buonaiuto 等人（2023）與 Carrascal 等人（2024）則在 IBM 真實裝置上執行 VQE 並做回測。整體而言，閘模型目前能處理的資產數比退火機更小，Stopfer 與 Wagner（2025）的結果也顯示 QAOA 在 20 檔以上時解品質迅速退化。

**綜合判斷。** 三波文獻呈現一致的軌跡：問題越貼近實務、對照組越強，量子方法的相對優勢就越小。截至目前，沒有任何研究在標準的均值—變異數問題上展示出穩健的量子優勢；可能的空間僅剩下古典求解器真正困難的設定，例如大規模基數限制、整數交易單位與非線性交易成本的組合。

---

## 5. 古典 baseline 與公平比較

衡量量子求解器的價值，前提是選對古典對照組。帶有基數、整數交易單位或交易成本的投資組合問題，在數學上通常是混合整數二次規劃（MIQP），已知為 NP-hard（Bienstock, 1996）。古典方法大致可分為以下五類：

| 類別 | 代表方法 | 解空間 | 限制處理 | 特性與適用情境 |
| :-- | :-- | :-- | :-- | :-- |
| 精確求解器 | 分支界限（branch and bound）、割平面（cutting planes）；Gurobi、CPLEX；針對基數限制的專屬演算法（Bonami & Lejeune, 2009; Bertsimas & Cory-Wright, 2022; Xu et al., 2024） | 連續、整數或混合 | 原生支援線性與二次限制 | 可證明全域最佳；中小規模極快，最壞情況下時間仍隨問題規模指數成長 |
| 凸最佳化 | 內點法、SQP；PyPortfolioOpt、CVXPY | 連續 | 原生支援 | 無整數限制的均值—變異數問題可在毫秒級解出全域最佳 |
| 啟發式與元啟發式 | 模擬退火（Kirkpatrick et al., 1983）、禁忌搜尋、遺傳演算法、粒子群（Chang et al., 2000） | 連續或離散 | 以懲罰或修復處理 | 不保證最佳，但可在合理時間內給出良好解；參數調校耗時 |
| QUBO 層級的古典求解器 | 模擬退火（`neal`）、禁忌搜尋、最陡下降（Kochenberger et al., 2014） | 二元 | 與 QPU 相同的懲罰形式 | 與量子退火解完全相同的問題，是衡量「量子取樣本身」價值的最直接對照 |
| 量子啟發（quantum-inspired） | Fujitsu Digital Annealer（Aramon et al., 2019）、張量網路（Mugel et al., 2022）、結合懲罰係數估計的量子啟發 QUBO 求解（Lu et al., 2024） | 二元 | 懲罰項 | 在古典硬體上模擬量子或物理過程；可處理比現有 QPU 更大的 QUBO |

精確求解器與窮舉法雖然都能保證全域最佳，運作邏輯卻截然不同。窮舉法逐一評估所有組合，例如從 50 檔中選 12 檔即有 $\binom{50}{12}\approx 1.2\times10^{11}$ 種組合；Gurobi 等求解器則以分支界限估計每個子樹的目標值界限，一旦界限劣於現有最佳解便整枝剪除，並以割平面在解空間中切除不可能包含最佳解的區域。這使得「精確求解」在實務規模下往往遠比直覺快速，也是 Stopfer 與 Wagner（2025）能在數秒內解出千檔實例的原因。

文獻對「公平比較」已有明確的方法論。Rønnow 等人（2014）指出，量子加速必須以「與最佳古典演算法在同一問題上的時間擴展（scaling）」來定義，而不能以單一實例或與次佳演算法比較來宣稱；Denchev 等人（2016）與 Albash 及 Lidar（2018b）也都是在精心構造的實例上，才觀察到退火機相對於模擬退火的擴展優勢。對投資組合研究而言，這意味著至少要遵守三項原則：(1) **解空間匹配**：評估 BQM 時，古典對照組也應在同一個 QUBO 上求解（例如 `neal`、禁忌搜尋），評估 CQM 或連續權重時，古典對照組應在連續或混合整數空間求解（例如 Gurobi、凸最佳化）；(2) **時間預算對等**：比較應在相同的牆鐘時間或計費時間下進行，並分開報告 QPU 存取時間與總時間（Lozano, 2026b）；(3) **多重指標**：同時報告解品質（regret、近似比、可行解比例）與投資績效，因為兩者不必然一致。

---

## 6. 文獻缺口與批判

逐篇閱讀上述文獻後，筆者認為現有研究在實驗設計上有四個共通的缺口。

### 6.1 回測架構不夠清晰，也不符合實務

投資組合最佳化的實務流程有其固定的要素，而多數量子投組研究只交代其中一部分。一個可被檢驗的回測設計至少應明確說明：(1) 資產宇宙為何、為何選這些資產、資料來源為何；(2) 樣本期間；(3) 單期或多期最佳化；(4) 納入哪些限制式；(5) 資料前處理步驟；(6) 預期報酬如何產生；(7) 共變異數矩陣如何估計；(8) 再平衡頻率及其理由。

以 Mugel 等人（2022）為例，其資產宇宙為 52 檔未具名的政府債券、固定收益與變動收益證券，作者並未說明選擇依據。更值得商榷的是其前處理流程：先以過去 20 個營業日的報酬刪除「低報酬且低變異」的資產，再對價格序列做 Hodrick–Prescott（HP）濾波取得趨勢項，以趨勢項分成 6–8 群，群內等權重，最後只對「是否買入某一群」做最佳化。筆者認為此流程有四個問題：

1. **刪除低報酬、低風險資產並不合理。** 若某檔資產與其他高報酬資產的相關性很低甚至為負，它在 $\sum_{i,j}w_iw_j\sigma_{ij}$ 中能大幅抵銷其他資產的波動，扮演避險或穩定器的角色，反而能提升整體的風險調整後報酬。以個別資產的報酬與風險預先篩選，忽略的正是均值—變異數框架最核心的共變異結構。
2. **債券不應以價格分群。** 債券價格會隨到期日推移而系統性變動，應改用含息總報酬 $R_t=(P_t-P_{t-1}+C_t)/P_{t-1}$。
3. **HP 濾波不適合用於分群。** 資產價格通常為非定態序列，HP 趨勢項仍是非定態的，以距離或相關係數衡量非定態序列極易產生假性相關；此外，HP 濾波在樣本端點的估計嚴重失真且會隨新資料修正（Hamilton, 2018），以最新一期的趨勢值決定下一期配置，會導致不必要的高週轉率。
4. **群內等權重喪失了最佳化的意義。** 最佳化只決定「買哪幾群」，群內權重固定，等於放棄了大部分可最佳化的自由度。

較穩健的替代做法包括：以對數報酬的相關係數矩陣做階層式分群，即階層式風險平價（HRP; López de Prado, 2016），並可搭配收縮或去雜訊；以主成分分析萃取隱含因子，依因子負荷分群（Acharya et al., 2025 亦採用隨機矩陣理論處理相關矩陣後再做譜分群）；或在必須對單一資產做時間序列平滑時，改用不使用未來資料的卡爾曼濾波（Kalman filter）。

在預期報酬方面，筆者認為兩種設定應同時檢驗，因為它們衡量的是不同的東西：以**下一期的實際報酬**作為輸入，可以純粹評估演算法找到的解的品質；以**機器學習模型的預測**作為輸入，則更貼近實務上會遇到的雜訊環境。現有研究多半只做其中一種，且以歷史平均報酬為主。

### 6.2 評估指標不一致，且往往只用單一指標

現有研究衡量演算法的方式大致分成三類，但多數只採用其中一類：

- **目標函數的遺憾值（objective regret）或近似比**：與「最佳可用的離線參考解」比較目標值差距。解空間小時（例如 24 檔取 12 檔，$\binom{24}{12}=2{,}704{,}156$ 種組合）可用窮舉法作參考，解空間大時則需仰賴精確求解器或貪婪演算法（Lozano, 2026a; Stopfer & Wagner, 2025）。
- **投資績效**：所得投組在一段期間內的報酬風險比（Mugel et al., 2022 只報告這一類，未衡量 regret）。
- **求解時間**。

筆者認為三類指標都應同時報告。投資績效方面至少包含累積報酬、報酬標準差、報酬風險比、最大回撤（MDD）、Calmar ratio（年化報酬／最大回撤）與交易成本；解品質方面則包含與參考解的 regret 與可行解比例；時間方面應區分建模時間、取樣時間與（若使用 QPU）QPU 存取時間。只有同時觀察這三個面向，才能判斷「解得比較好」是否真的轉化為「投資比較好」。

### 6.3 BQM、DQM、CQM 從未在同一框架下被公平比較

現有研究幾乎都只選用 BQM 或只選用 CQM 其中之一，因此無法回答「模型框架的選擇本身」對結果的影響。更重要的是，對照組的解空間往往與待測求解器不一致：拿離散化的 QUBO 解與連續空間的凸最佳化解比較，差異究竟來自求解器，還是來自離散化本身，便無從分辨。公平的設計應讓古典演算法與量子求解器面對同一個問題：比較 BQM 時，古典演算法同樣求解 QUBO；比較 CQM 時，古典演算法則在連續或混合整數空間中求解。

### 6.4 單一回測窗口

多數研究只在單一樣本期間（甚至單一時點）評估投組績效。金融報酬的結構隨時間變化，單一窗口的勝負很可能只是特定市場事件的產物。例如一個涵蓋 2025 年 3 月至 6 月的窗口，恰好包含 4 月的關稅衝擊與隨後的急速反彈，任何關於「哪個求解器樣本外較佳」的結論，都必須先排除這種單一事件的影響。滾動式（walk-forward）多期回測因此是任何績效主張的必要條件。

---

## 7. 本研究定位

### 7.1 研究問題

文獻的評估範式幾乎都以 regret 為核心：以精確求解器的最佳解為標竿，量子方法差距越小越好。這隱含了「目標函數最佳即投資最佳」的假設。然而 Markowitz 框架的估計誤差文獻早已指出，最佳化器會放大輸入誤差，「最佳化的」不等於「最好的」（Michaud, 1989; Chopra & Ziemba, 1993），而對解空間施加限制反而可能改善樣本外表現（Jagannathan & Ma, 2003; Behr et al., 2013）。由此可推出一個在量子投組文獻中尚未被檢驗的假說：當預期報酬來自含雜訊的機器學習預測時，精確的 MSRP 會集中押注被高估的股票；離散化的 QUBO 解則可能因權重格點與不完全的搜尋而被迫分散，樣本外表現反而較佳。

因此本研究的核心問題是：**在含雜訊的輸入下，求解器的解品質與投資績效是否脫鉤？權重離散化是否扮演隱式正則化的角色？**

### 7.2 研究設計如何回應文獻缺口

| 文獻缺口 | 本研究的設計 |
| :-- | :-- |
| 資產宇宙與選擇理由不明（6.1） | 以台灣 50（0050）ETF 成分股為資產宇宙（約 50 檔），成分股名單公開、流動性高，且為台灣投資人最常見的被動投資標的 |
| 再平衡頻率缺乏理由（6.1） | 於 0050 每季成分股調整公告日（3、6、9、12 月第一個星期五）收盤決定權重，持有至下一次公告日，與指數本身的換股節奏一致 |
| 預期報酬來源單一（6.1） | 以合併訓練（pooled）的單層 LSTM（Hochreiter & Schmidhuber, 1997）預測未來一季報酬，訓練資料嚴格切在公告日之前並設置隔離區間以避免資訊洩漏；另以事後實現的報酬與共變異數求出 oracle 權重，僅作為診斷基準 |
| 共變異數估計單一（6.1） | 同時比較樣本共變異數、EWMA，以及 Ledoit–Wolf 三種收縮目標（單因子、固定相關係數、單位矩陣）（Ledoit & Wolf, 2003, 2004a, 2004b） |
| 對照組解空間不一致（6.3） | 四類求解器面對同一個均值—變異數目標：連續空間的凸最佳化（PyPortfolioOpt）、連續空間的模擬退火（`scipy` dual annealing）、QUBO 的模擬退火（D-Wave Ocean `neal`，$P=20$ 與 $P=100$），以及將凸最佳化解以最大餘數法四捨五入到同一格點的古典離散化基準，藉此把「離散化效果」與「求解器效果」分開 |
| 懲罰係數須人工調整（3.2） | 以「每單位違反所能換到的最大目標改善量」$L$ 作為 $\lambda$ 的尺度，並以只使用事前輸入的敏感度實驗（$\lambda/L$ × sweeps）選定係數；同時記錄每期的可行解比例與修復比例 |
| 離散化效果與正則化效果混淆（6.3、8） | 另加入「MVO + 10% 單一權重上限」作為顯式正則化對照組，檢驗退火解的分散效果能否被簡單限制複製 |
| 評估指標單一（6.2） | 同時報告解品質（與連續最佳解的目標值差距、權重距離）與投資績效（累積報酬、波動度、夏普、MDD、Calmar、扣除手續費 0.1425% 與證交稅 0.3% 後的淨值），以及求解時間 |
| 單一回測窗口（6.4） | 2025 年 3 月至 2026 年 3 月共 5 季的滾動式樣本外回測 |

### 7.3 主要發現摘要

5 季 walk-forward 的完整結果見 repo 首頁的 [README](../README.md#主要結果)，與本章論點相關的發現有三項：

1. **解品質的瓶頸在懲罰地形，而非離散化。** 在相同目標函數下，QUBO＋模擬退火每季損失約 1～4 個百分點的確定等值報酬，但把凸最佳解四捨五入到同一格點幾乎沒有損失；懲罰係數越小解越好，增加 sweeps 幾乎無助。這與 Lozano（2026a）在 QPU 上觀察到的「瓶頸在懲罰編碼」一致，且顯示同樣的現象在古典退火上也存在。
2. **解品質與投資績效確實脫鉤，但脫鉤的來源是 beta 與分散度。** 退火解較分散、對大盤的 beta 較高，在研究期間的多頭行情中樣本外 Sharpe 高於精確解；然而以 GMVP 的本意（低波動）衡量，精確解仍較佳，MSRP 的優勢則可被 10% 權重上限複製。因此，「離散化即隱式正則化」的假說只得到弱支持：真正起作用的是次佳解的分散，而這可以用古典的顯式限制以更低成本達成。
3. **被動基準大幅領先。** 等權重與 0050 ETF 的 Sharpe 均遠高於所有最佳化組合，呼應 DeMiguel 等人（2009）的發現。

### 7.4 研究限制

本研究的定位必須誠實界定。第一，**所有 QUBO 結果均以 D-Wave Ocean 的古典模擬退火（`neal`）產生，並未使用真實 QPU 或 hybrid solver**；程式已保留切換至 QPU 與 Leap hybrid 的介面，但 50 檔 × 7 位元約 350 個變數的稠密 QUBO 已超過 Advantage 可直接嵌入的完全圖規模，實際上機需縮減問題或改用 hybrid solver。因此本研究的結論是關於「離散化的 QUBO 形式」而非「量子效應」。第二，資產數僅約 50 檔，且 long-only 均值—變異數問題本身是凸問題，古典求解器可在毫秒內解出全域最佳，本研究並不主張任何計算上的量子優勢。第三，5 季的樣本外期間仍然有限，且全部落在台股的強勁多頭行情（0050 上漲 128%），統計檢定力不足以區分 beta 與 alpha，結論應視為對「隱式正則化」假說的初步證據。第四，QUBO 版本的 MSRP 是以掃描風險趨避係數 $\gamma$ 並取事前夏普最高者近似，與直接最大化夏普比率的凸最佳化解並非完全相同的問題，比較時需注意此一差異。

---

## 8. 未來研究方向

綜合文獻現況與本研究的限制，筆者認為後續研究有四個方向：

1. **以顯式正則化檢驗隱式正則化。** 若離散化確實發揮正則化作用，則應檢驗它能否被「凸最佳化 + 低成本的顯式正則化」複製或超越，例如權重上限、$L_2$ 懲罰、收縮估計、穩健最佳化或重抽樣效率前緣（Michaud, 1989）。進一步可對真實報酬注入不同強度的雜訊，描繪「輸入雜訊強度 vs. 各求解器樣本外表現」的曲線，找出離散解開始勝出的門檻。無論結果為何，都能釐清 QUBO 形式在金融上的實際價值：若能被複製，所謂的穩健性只是正則化的副產品；若不能，則離散化本身具有獨特貢獻。這個方向不需要量子硬體即可完成。
2. **走向古典求解器真正困難的問題。** 文獻顯示，量子優勢若存在，只可能出現在古典方法開始失效之處：基數限制、整數交易單位（台股以「張」為單位）、階梯式手續費與多期動態的組合。具體做法是先在台股全市場的「$N$ 檔選 $K$ 檔 + 整張交易 + 階梯成本」設定下確認 Gurobi 的求解時間確實開始急遽成長，避免再度落入「古典其實秒解」的陷阱，再測試 hybrid CQM 與無懲罰管線（Lozano, 2026a）。台灣市場高度集中於電子產業（0050 的權重以半導體與電子類股為主），其相關結構也與既有文獻集中的美、日、歐市場不同。
3. **真實 QPU 與 hybrid solver 的實測。** 在縮減後的問題上以 Advantage／Advantage2 QPU 直接求解，並以相同時間預算比較 `neal`、禁忌搜尋與 hybrid BQM／CQM，同時記錄 QPU 存取時間、鏈斷裂率與可行解比例，以 BQM、DQM、CQM 三種框架在同一問題上的完整比較回應 6.3 節的缺口。
4. **交易成本與更長的樣本期間。** 將交易成本與週轉率限制直接納入目標函數（而非只在回測時扣除），近期已有研究在 CQM 中處理此類多期再平衡問題（Sánchez-Martínez et al., 2026），並把樣本外期間延長至涵蓋多種市場狀態，以提高結論的統計可信度。

---

## 9. 參考文獻

> 所有文獻皆已以 Crossref、arXiv 或出版者頁面核對題名、作者、年份與出處。僅以 arXiv 發表者標示 arXiv 編號。

Acharya, A., Yalovetzky, R., Minssen, P., Chakrabarti, S., Shaydulin, R., Raymond, R., Sun, Y., Herman, D., Andrist, R. S., Salton, G., Schuetz, M. J. A., Katzgraber, H. G., & Pistoia, M. (2025). Decomposition pipeline for large-scale portfolio optimization with applications to near-term quantum computing. *Physical Review Research, 7*(2), 023142. https://doi.org/10.1103/PhysRevResearch.7.023142

Albash, T., & Lidar, D. A. (2018a). Adiabatic quantum computation. *Reviews of Modern Physics, 90*(1), 015002. https://doi.org/10.1103/RevModPhys.90.015002

Albash, T., & Lidar, D. A. (2018b). Demonstration of a scaling advantage for a quantum annealer over simulated annealing. *Physical Review X, 8*(3), 031016. https://doi.org/10.1103/PhysRevX.8.031016

Arai, S., Oshiyama, H., & Nishimori, H. (2023). Effectiveness of quantum annealing for continuous-variable optimization. *Physical Review A, 108*(4), 042403. https://doi.org/10.1103/PhysRevA.108.042403

Aramon, M., Rosenberg, G., Valiante, E., Miyazawa, T., Tamura, H., & Katzgraber, H. G. (2019). Physics-inspired optimization for quadratic unconstrained problems using a digital annealer. *Frontiers in Physics, 7*, 48. https://doi.org/10.3389/fphy.2019.00048

Barkoutsos, P. Kl., Nannicini, G., Robert, A., Tavernelli, I., & Woerner, S. (2020). Improving variational quantum optimization using CVaR. *Quantum, 4*, 256. https://doi.org/10.22331/q-2020-04-20-256

Behr, P., Guettler, A., & Miebs, F. (2013). On portfolio optimization: Imposing the right constraints. *Journal of Banking & Finance, 37*(4), 1232–1242. https://doi.org/10.1016/j.jbankfin.2012.11.020

Bertsimas, D., & Cory-Wright, R. (2022). A scalable algorithm for sparse portfolio selection. *INFORMS Journal on Computing, 34*(3), 1489–1511. https://doi.org/10.1287/ijoc.2021.1127

Bienstock, D. (1996). Computational study of a family of mixed-integer quadratic programming problems. *Mathematical Programming, 74*(2), 121–140. https://doi.org/10.1007/BF02592208

Bonami, P., & Lejeune, M. A. (2009). An exact solution approach for portfolio optimization problems under stochastic and integer constraints. *Operations Research, 57*(3), 650–670. https://doi.org/10.1287/opre.1080.0599

Boothby, K., Bunyk, P., Raymond, J., & Roy, A. (2020). *Next-generation topology of D-Wave quantum processors* (arXiv:2003.00133). arXiv. https://doi.org/10.48550/arXiv.2003.00133

Brandhofer, S., Braun, D., Dehn, V., Hellstern, G., Hüls, M., Ji, Y., Polian, I., Bhatia, A. S., & Wellens, T. (2022). Benchmarking the performance of portfolio optimization with QAOA. *Quantum Information Processing, 22*(1), 25. https://doi.org/10.1007/s11128-022-03766-5

Buonaiuto, G., Gargiulo, F., De Pietro, G., Esposito, M., & Pota, M. (2023). Best practices for portfolio optimization by quantum computing, experimented on real quantum devices. *Scientific Reports, 13*, 19434. https://doi.org/10.1038/s41598-023-45392-w

Carrascal, G., Hernamperez, P., Botella, G., & del Barrio, A. (2024). Backtesting quantum computing algorithms for portfolio optimization. *IEEE Transactions on Quantum Engineering, 5*, 1–20. https://doi.org/10.1109/TQE.2023.3337328

Chang, T.-J., Meade, N., Beasley, J. E., & Sharaiha, Y. M. (2000). Heuristics for cardinality constrained portfolio optimisation. *Computers & Operations Research, 27*(13), 1271–1302. https://doi.org/10.1016/S0305-0548(99)00074-X

Chopra, V. K., & Ziemba, W. T. (1993). The effect of errors in means, variances, and covariances on optimal portfolio choice. *The Journal of Portfolio Management, 19*(2), 6–11. https://doi.org/10.3905/jpm.1993.409440

Cohen, J., Khan, A., & Alexander, C. (2020a). *Portfolio optimization of 40 stocks using the DWave quantum annealer* (arXiv:2007.01430). arXiv. https://doi.org/10.48550/arXiv.2007.01430

Cohen, J., Khan, A., & Alexander, C. (2020b). *Portfolio optimization of 60 stocks using classical and quantum algorithms* (arXiv:2008.08669). arXiv. https://doi.org/10.48550/arXiv.2008.08669

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/*N* portfolio strategy? *The Review of Financial Studies, 22*(5), 1915–1953. https://doi.org/10.1093/rfs/hhm075

Denchev, V. S., Boixo, S., Isakov, S. V., Ding, N., Babbush, R., Smelyanskiy, V., Martinis, J., & Neven, H. (2016). What is the computational value of finite-range tunneling? *Physical Review X, 6*(3), 031015. https://doi.org/10.1103/PhysRevX.6.031015

D-Wave Quantum Inc. (n.d.). *CQM solver properties*. D-Wave Quantum Computing Products Documentation. Retrieved October 5, 2026, from https://docs.dwavequantum.com/en/latest/industrial_optimization/solver_cqm_properties.html

Egger, D. J., Gambella, C., Marecek, J., McFaddin, S., Mevissen, M., Raymond, R., Simonetto, A., Woerner, S., & Yndurain, E. (2020). Quantum computing for finance: State-of-the-art and future prospects. *IEEE Transactions on Quantum Engineering, 1*, 1–24. https://doi.org/10.1109/TQE.2020.3030314

Fernández-Lorenzo, S., Porras, D., & García-Ripoll, J. J. (2021). Hybrid quantum–classical optimization with cardinality constraints and applications to finance. *Quantum Science and Technology, 6*(3), 034010. https://doi.org/10.1088/2058-9565/abf9af

Glover, F., Kochenberger, G., & Du, Y. (2019). Quantum Bridge Analytics I: A tutorial on formulating and using QUBO models. *4OR, 17*(4), 335–371. https://doi.org/10.1007/s10288-019-00424-y

Grant, E., Humble, T. S., & Stump, B. (2021). Benchmarking quantum annealing controls with portfolio optimization. *Physical Review Applied, 15*(1), 014012. https://doi.org/10.1103/PhysRevApplied.15.014012

Hamilton, J. D. (2018). Why you should never use the Hodrick-Prescott filter. *The Review of Economics and Statistics, 100*(5), 831–843. https://doi.org/10.1162/rest_a_00706

Hauke, P., Katzgraber, H. G., Lechner, W., Nishimori, H., & Oliver, W. D. (2020). Perspectives of quantum annealing: Methods and implementations. *Reports on Progress in Physics, 83*(5), 054401. https://doi.org/10.1088/1361-6633/ab85b8

Herman, D., Googin, C., Liu, X., Sun, Y., Galda, A., Safro, I., Pistoia, M., & Alexeev, Y. (2023). Quantum computing for finance. *Nature Reviews Physics, 5*(8), 450–465. https://doi.org/10.1038/s42254-023-00603-1

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Hodson, M., Ruck, B., Ong, H., Garvin, D., & Dulman, S. (2019). *Portfolio rebalancing experiments using the Quantum Alternating Operator Ansatz* (arXiv:1911.05296). arXiv. https://doi.org/10.48550/arXiv.1911.05296

Jagannathan, R., & Ma, T. (2003). Risk reduction in large portfolios: Why imposing the wrong constraints helps. *The Journal of Finance, 58*(4), 1651–1683. https://doi.org/10.1111/1540-6261.00580

Kadowaki, T., & Nishimori, H. (1998). Quantum annealing in the transverse Ising model. *Physical Review E, 58*(5), 5355–5363. https://doi.org/10.1103/PhysRevE.58.5355

Kerenidis, I., Prakash, A., & Szilágyi, D. (2019). Quantum algorithms for portfolio optimization. In *Proceedings of the 1st ACM Conference on Advances in Financial Technologies* (pp. 147–155). ACM. https://doi.org/10.1145/3318041.3355465

Kim, S., Ahn, S.-W., Suh, I.-S., Dowling, A. W., Lee, E., & Luo, T. (2025). Quantum annealing for combinatorial optimization: A benchmarking study. *npj Quantum Information, 11*, 77. https://doi.org/10.1038/s41534-025-01020-1

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science, 220*(4598), 671–680. https://doi.org/10.1126/science.220.4598.671

Kochenberger, G., Hao, J.-K., Glover, F., Lewis, M., Lü, Z., Wang, H., & Wang, Y. (2014). The unconstrained binary quadratic programming problem: A survey. *Journal of Combinatorial Optimization, 28*(1), 58–81. https://doi.org/10.1007/s10878-014-9734-0

Lang, J., Zielinski, S., & Feld, S. (2022). Strategic portfolio optimization using simulated, digital, and quantum annealing. *Applied Sciences, 12*(23), 12288. https://doi.org/10.3390/app122312288

Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *Journal of Empirical Finance, 10*(5), 603–621. https://doi.org/10.1016/S0927-5398(03)00007-0

Ledoit, O., & Wolf, M. (2004a). Honey, I shrunk the sample covariance matrix. *The Journal of Portfolio Management, 30*(4), 110–119. https://doi.org/10.3905/jpm.2004.110

Ledoit, O., & Wolf, M. (2004b). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis, 88*(2), 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4

López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *The Journal of Portfolio Management, 42*(4), 59–69. https://doi.org/10.3905/jpm.2016.42.4.059

Lozano, L. (2026a). *A penalty-free pipeline for direct quantum-annealer portfolio optimization* (arXiv:2605.17628). arXiv. https://doi.org/10.48550/arXiv.2605.17628

Lozano, L. (2026b). *Where the quantum lives in D-Wave hybrid portfolio optimization: An operational decomposition audit* (arXiv:2605.17623). arXiv. https://doi.org/10.48550/arXiv.2605.17623

Lu, Y.-C., Fu, C.-M., Yu, L.-P., Chang, Y.-J., & Chang, C.-R. (2024). *Quantum-inspired portfolio optimization in the QUBO framework* (arXiv:2410.05932). arXiv. https://doi.org/10.48550/arXiv.2410.05932

Lucas, A. (2014). Ising formulations of many NP problems. *Frontiers in Physics, 2*, 5. https://doi.org/10.3389/fphy.2014.00005

Markowitz, H. (1952). Portfolio selection. *The Journal of Finance, 7*(1), 77–91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x

Mattesi, M., Asproni, L., Mattia, C., Tufano, S., Ranieri, G., Caputo, D., & Corbelletto, D. (2023). *Diversifying investments and maximizing Sharpe ratio: A novel QUBO formulation* (arXiv:2302.12291). arXiv. https://doi.org/10.48550/arXiv.2302.12291

Michaud, R. O. (1989). The Markowitz optimization enigma: Is "optimized" optimal? *Financial Analysts Journal, 45*(1), 31–42. https://doi.org/10.2469/faj.v45.n1.31

Morapakula, S. N., Deshpande, S., Yata, R., Ubale, R., Wad, U., & Ikeda, K. (2026). End-to-end portfolio optimization with hybrid quantum annealing. *Advanced Quantum Technologies, 9*(4), e00753. https://doi.org/10.1002/qute.202500753

Mugel, S., Abad, M., Bermejo, M., Sánchez, J., Lizaso, E., & Orús, R. (2021). Hybrid quantum investment optimization with minimal holding period. *Scientific Reports, 11*, 19587. https://doi.org/10.1038/s41598-021-98297-x

Mugel, S., Kuchkovsky, C., Sánchez, E., Fernández-Lorenzo, S., Luis-Hita, J., Lizaso, E., & Orús, R. (2022). Dynamic portfolio optimization with real datasets using quantum processors and quantum-inspired tensor networks. *Physical Review Research, 4*(1), 013006. https://doi.org/10.1103/PhysRevResearch.4.013006

Orús, R., Mugel, S., & Lizaso, E. (2019). Quantum computing for finance: Overview and prospects. *Reviews in Physics, 4*, 100028. https://doi.org/10.1016/j.revip.2019.100028

Palmer, S., Karagiannis, K., Florence, A., Rodriguez, A., Orús, R., Naik, H., & Mugel, S. (2022). *Financial index tracking via quantum computing with cardinality constraints* (arXiv:2208.11380). arXiv. https://doi.org/10.48550/arXiv.2208.11380

Palmer, S., Sahin, S., Hernandez, R., Mugel, S., & Orús, R. (2021). *Quantum portfolio optimization with investment bands and target volatility* (arXiv:2106.06735). arXiv. https://doi.org/10.48550/arXiv.2106.06735

Phillipson, F., & Bhatia, H. S. (2021). Portfolio optimisation using the D-Wave quantum annealer. In *Computational Science – ICCS 2021* (Lecture Notes in Computer Science, pp. 45–59). Springer. https://doi.org/10.1007/978-3-030-77980-1_4

Rebentrost, P., & Lloyd, S. (2024). Quantum computational finance: Quantum algorithm for portfolio optimization. *KI – Künstliche Intelligenz, 38*, 327–338. https://doi.org/10.1007/s13218-024-00870-9

Rønnow, T. F., Wang, Z., Job, J., Boixo, S., Isakov, S. V., Wecker, D., Martinis, J. M., Lidar, D. A., & Troyer, M. (2014). Defining and detecting quantum speedup. *Science, 345*(6195), 420–424. https://doi.org/10.1126/science.1252319

Rosenberg, G., Haghnegahdar, P., Goddard, P., Carr, P., Wu, K., & López de Prado, M. (2016). Solving the optimal trading trajectory problem using a quantum annealer. *IEEE Journal of Selected Topics in Signal Processing, 10*(6), 1053–1060. https://doi.org/10.1109/JSTSP.2016.2574703

Sakuler, W., Oberreuter, J. M., Aiolfi, R., Asproni, L., Roman, B., & Schiefer, J. (2025). A real-world test of portfolio optimization with quantum annealing. *Quantum Machine Intelligence, 7*(1), 43. https://doi.org/10.1007/s42484-025-00268-2

Sánchez-Martínez, E., Hernandez Santana, S., Sarasa Laborda, V., Serrano Molinero, P., Botella Juan, G., & del Barrio García, A. (2026). *Quantum annealing for dynamic portfolio optimization under realistic transaction costs* (arXiv:2607.03218). arXiv. https://doi.org/10.48550/arXiv.2607.03218

Stopfer, E., & Wagner, F. (2025). *Quantum portfolio optimization: An extensive benchmark* (arXiv:2509.17876). arXiv. https://doi.org/10.48550/arXiv.2509.17876

Tamura, K., Shirai, T., Katsura, H., Tanaka, S., & Togawa, N. (2021). Performance comparison of typical binary-integer encodings in an Ising machine. *IEEE Access, 9*, 81032–81039. https://doi.org/10.1109/ACCESS.2021.3081685

Venturelli, D., & Kondratyev, A. (2019). Reverse quantum annealing approach to portfolio optimization problems. *Quantum Machine Intelligence, 1*(1–2), 17–30. https://doi.org/10.1007/s42484-019-00001-w

Verma, A., & Lewis, M. (2022). Penalty and partitioning techniques to improve performance of QUBO solvers. *Discrete Optimization, 44*, 100594. https://doi.org/10.1016/j.disopt.2020.100594

Wei, H., Wang, Y. J., Yang, H., Yang, X., Cao, M., Xu, Q., Cai, M., Wang, Y., Mao, Z., Cao, X., Mei, Q., Wang, J., Zhou, X., Yao, L., & Zhao, W. (2026). Solving multiple discretization portfolio optimization problem with quantum-classical hybrid algorithms. *Computational Economics, 68*(1), 227–256. https://doi.org/10.1007/s10614-025-11061-5

Xu, H., Dasgupta, S., Pothen, A., & Banerjee, A. (2023). Dynamic asset allocation with expected shortfall via quantum annealing. *Entropy, 25*(3), 541. https://doi.org/10.3390/e25030541

Xu, W., Tang, J., Yiu, K. F. C., & Peng, J. W. (2024). An efficient global optimal method for cardinality constrained portfolio optimization. *INFORMS Journal on Computing, 36*(2), 690–704. https://doi.org/10.1287/ijoc.2022.0344
