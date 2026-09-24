# 相关工作文献检索：TTA / 读出侧适配（lit_search_tta）

> 出处：claude-0b 派活（2026-09-24）。**为论文 §1.x 相关工作补弹药**，
> 起因是外部复核者 §丁 D2 的风险提示：审稿人更可能来自 **TTA / 域适应社群**，
> 而本项目**没有引用任何 TTA / 头重校准方向的文献**，
> ⇒「**这不就是 target-domain 的 last-layer 重训吗**」会成为一个没有引文可答的问题。
>
> **纪律（本项目刚在 `nat` 上栽过一次）**：每条候选都给**可点开的证据**，
> 并用 **OpenAlex / Crossref / arXiv 三套 API 之一逐字核**；
> **三分法**（找到 / 找不到 / 检索能力不足）；**不许把"找不到"说成"不存在"**；
> **宁少勿软**。**未动 `paper/**`。**

---

## 方向 1：TTA / TTT（含"只更新最后一层 / 只更新归一化参数"那一支）

全部经 **OpenAlex 逐字核**（标题/作者/年份/出处），其中两条另经 **Crossref 交叉核**。

**[1] Tent —— 只更新 BN 仿射参数**
> Wang D, Shelhamer E, Liu S, et al. **Tent: Fully Test-time Adaptation by Entropy Minimization**.
> arXiv preprint, 2020. arXiv:**2006.10726** · DOI: https://doi.org/10.48550/arxiv.2006.10726
> ✅ 已核（OpenAlex：标题✓ 作者✓ 年份✓ 出处✓）。**注**：OpenAlex 记为 arXiv 记录；
> 若要著录 ICLR 2021，需另行确认（本次未核到 ICLR 的 DOI）。
> 与本案关系：**"冻结主干 + 只动极少数参数"在 TTA 社群里是标准做法**；本项目把可动参数进一步压到**只有一个线性头**。

**[2] TTT —— 测试时自监督**
> Sun Y, Wang X, Liu Z, et al. **Test-Time Training with Self-Supervision for Generalization under
> Distribution Shifts**. arXiv preprint, 2019. arXiv:**1909.13231** · DOI: https://doi.org/10.48550/arxiv.1909.13231
> ✅ 已核。与本案关系：分布偏移下"测试时更新"的开山；**但它更新的是主干**，本项目**物理上不能**。

**[3] CoTTA —— 连续/持续 TTA**
> Wang Q, Fink O, Van Gool L, et al. **Continual Test-Time Domain Adaptation**.
> CVPR 2022: 7191–7201. DOI: https://doi.org/10.1109/cvpr52688.2022.00706
> ✅ 已核（含页码）。与本案关系：TTA 的**遗忘/误差累积**问题——本项目读出适配**离线一次**，不面临该问题。

**[4] MEMO —— 测试时增强 + 边缘熵**
> Zhang M, Levine S, Finn C. **MEMO: Test Time Robustness via Adaptation and Augmentation**.
> arXiv preprint, 2021. arXiv:**2110.09506** · DOI: https://doi.org/10.48550/arxiv.2110.09506
> ✅ 已核。

**[5] 预测试时 BN（只重估归一化统计量）**
> Nado Z, Padhy S, Sculley D, et al. **Evaluating Prediction-Time Batch Normalization for
> Robustness under Covariate Shift**. arXiv preprint, 2020. arXiv:**2006.10963**
> ✅ 已核（OpenAlex）。**注**：该 arXiv DOI **未在 Crossref 注册**（404），故只有单源核验。
> 与本案关系：**最便宜的"读出侧"适配**（零训练、只重估统计量）。

**[6] 协变量漂移适配**
> Schneider S, Rusak E, Eck L, et al. **Improving robustness against common corruptions by
> covariate shift adaptation**. NeurIPS 2020, **33**: 11539–11551. arXiv:**2006.16971**
> ✅ 已核（OpenAlex 同时给出 NeurIPS 与 arXiv 两条记录，标题/作者一致）。

**[7] NOTE —— 持续 TTA 抗时间相关**
> Gong T, Jeong J, Kim T, et al. **NOTE: Robust Continual Test-time Adaptation Against Temporal
> Correlation**. NeurIPS 2022: 27253–27266. arXiv:**2208.05117**
> ✅ 已核。

**[8] SHOT —— 只动分类头 / 不访问源数据**
> Liang J, Hu D, Feng J. **Do We Really Need to Access the Source Data? Source Hypothesis Transfer
> for Unsupervised Domain Adaptation**. arXiv preprint, 2020. arXiv:**2002.08546**
> ✅ 已核。**与本案最接近的一条**：**冻结特征提取器、只学分类头、且不需要源数据**。

**小结（方向 1）**：这一支的现状是——**"冻结主干 + 只更新极少数参数"在 TTA 社群已是标准范式（Tent 只动 BN 参数、SHOT 只动头）**。
本项目要答的问题**不是**"能不能只动读出"，而是：
**在 CIM 场景下主干物理上不可重训时，"只动读出"能救回多少、上界在哪、代价是什么**。
⇒ **引文可答**：「SHOT/Tent 已经证明冻结主干+轻量适配有效；本项目给出的是该策略在**权重不可重训**约束下的**上界与机制**」。

---

## 方向 2：last-layer recalibration / head recalibration / target-domain last-layer retraining

**[结论：本方向没有以这些词为标题的奠基文献；相关工作都挂在 TTA / DA 名下]**

- 用 `last layer recalibration domain adaptation classifier` 等检索式查 OpenAlex，返回的是**无关**条目
  （Domain Conditioned Adaptation Network、Squeeze-and-Excitation、旋转机械故障诊断…）。
- **这一支的代表作其实就是 [8] SHOT**（freeze feature extractor + 只学分类头 + 拒绝源数据）。
- ⇒ **不是空白，是"命名不同"**：**该方向以 TTA / source-free DA 的名字存在**。
  **建议论文用 SHOT / Tent 作为该方向的引文，不要另立"last-layer recalibration"这个说法**
  （那个说法在文献里不通用，会显得像是没做过检索）。

---

## 方向 3：Kumar et al.（linear probing under shift）**之后**的工作

**[1] 基线本身（本项目已核实）**
> Kumar A, Raghunathan A, Jones R, Ma T, Liang P. **Fine-Tuning can Distort Pretrained Features and
> Underperform Out-of-Distribution**. ICLR 2022. arXiv:**2202.10054**
> ✅ 已核（OpenAlex：被引 **160**）。

**[⚠ 未完成的检索 —— 如实记为"检索能力不足"，不是"没有后续"]**

- 我用 OpenAlex 的 `filter=cites:<W4221149036>` 拉它的施引文献（按被引排序前 8），
  结果**全是应用向**（医学影像微调策略比较、风电预测、日志异常检测、视网膜基础模型…），
  **没有方法学后续**。
- **判断**：OpenAlex 索引的是**这条 arXiv 预印本记录**，其引文图**明显不完整**
  （160 条里前面全是应用论文，方法学后续不在其中）。
- ⇒ **这是"我这条路径没走通"，不是"Kumar 之后没有方法学工作"。**
  **要补这一支，需要**：查 ICLR 2022 版本的独立记录，或用 Semantic Scholar 的引文图
  （本次它返回 **429 限流**）。

---

## 方向 4：CiM / 模拟域非理想性下的**读出侧**适配

**[结论：初步判断这一支是空白 —— 但这是"我没找到"，不是"不存在"]**

**找到的最近工作（都已核）**

> **[A]** Rasch M J, Mackin C, Le Gallo M, et al. **Hardware-aware training for large-scale and
> diverse deep learning inference workloads using in-memory computing**. *Nature Communications*,
> 2023, **14**(1): 5282. DOI: https://doi.org/10.1038/s41467-023-40770-4
> ✅ **已核（OpenAlex + Crossref 双源，标题/作者/卷/期/年逐字一致）**
>
> **[B]** Xiao Z, Naik V B, Lim J H, et al. **Adapting magnetoresistive memory devices for accurate
> and on-chip-training-free in-memory computing**. *Science Advances*, 2024, **10**(38): eadp3710.
> DOI: https://doi.org/10.1126/sciadv.adp3710
> ✅ **已核（OpenAlex + Crossref 双源，逐字一致）**

**但它们做的不是本项目的这件事**：
- **[A] 是训练端 HWA**（把非理想性建进训练），不是"冻结阵列 + 事后数字域适配"；
- **[B] 是器件级校准**（查找表 + 自适应量化，重新参数化预训练模型），**接近但不同**——
  它动的是**器件/量化层**，本项目动的是**读出分类头**，且本项目给出的是**上界（headroom）与机制（秩塌陷）**。

**依据（我查了什么）**：OpenAlex / Crossref 的 5 条检索式
（`compute-in-memory nonideality calibration readout adaptation` 等）
+ `LITERATURE_NOTES.md` §二 的 7 条 ⚠ 清单。
**最近的先行工作就是上面 [A][B] 两条**，没有出现"冻结阵列 + 数字域读出头重训 + 给出可恢复上界"的研究。
⇒ **这与 `docs/POSITIONING.md` 的定位一致，可作正面证据。**

**⚠ 边界**：以上是**"我在这些库里、用这些检索式没找到"**。
**不构成"不存在"的证明**（CiM 会议如 IMW/ISCAS 的正文可能未被这些库完整索引）。

---

## 🔔 附带收获：`LITERATURE_NOTES.md` §二 的 ⚠ 清单

**两条 ⚠ 已可升级为 ✅（双源核实）**

| §二 原条目 | 核实结果 |
|---|---|
| IBM *Hardware-aware training…*（Nat Commun 2023, s41467-023-40770-4） | ✅ Rasch M J, Mackin C, Le Gallo M, et al. **Nature Communications, 2023, 14(1): 5282** ✓ 标题/作者/卷/期/年逐字一致 |
| *Adapting magnetoresistive memory devices…*（Science Advances, DOI 10.1126/sciadv.adp3710） | ✅ Xiao Z, Naik V B, Lim J H, et al. **Science Advances, 2024, 10(38): eadp3710** ✓ 逐字一致 |

**⚠ 但发现 §二 有一处标题与编号不匹配（`nat` 同类问题，量级小得多）**

> §二 原文记：「***Less is More: On the Feature Redundancy of Pretrained Models*（arXiv 2310.03843）**」

- **arXiv 2310.03843 的实际标题**（取自 arXiv abs 页与 OpenAlex 两处）：
  **「From Channel Bias to Feature Redundancy: Uncovering the "Less is More" Principle in Few-Shot
  Learning」** —— 作者 Ji Zhang, Xu Luo, Lianli Gao, Difan Zou, Hengtao Shen, Jingkuan Song
  （submitted 2023-10-05 v1；last revised 2025-09-10 v2）。
- **但 §二 记录的内容摘要与真实摘要一致** ✓：
  "as few as **1-5%** of the most discriminative feature dimensions"、
  "**diminishing as more samples become available**" ⇔ §二 写"仅用 1% 的特征维度""冗余随样本数增加而消失"。
- ⇒ **编号对、内容对、标题对不上**。最可能是 **v1 的旧标题**，或转引时混入。
  **按红线 9，用之前应改用真实标题。**

---

## 四件套

- **里程碑**：相关工作文献检索（TTA / 读出侧适配）
- **关键数字 + 出处**：
  - **8 条** TTA/SHOT 方向候选，**全部经 OpenAlex 逐字核**，其中 2 条另有 Crossref 交叉核
  - **2 条** §二 ⚠ 升级为 ✅（Rasch 2023 / Xiao 2024，**双源**）
  - **1 条** 标题-编号不匹配（§二 的 arXiv 2310.03843）
- **判据命中哪个分支**：本任务无分支判据。按三分法：
  **方向1 = 找到（8 条硬证据）**；**方向2 = 找到（以 TTA/DA 之名存在，非空白）**；
  **方向3 = 检索能力不足（OpenAlex 引文图不完整 + Semantic Scholar 429）**；
  **方向4 = 找不到（初步判断为空白，但不断言不存在）**
- **有没有触发红线**：**没有**。**未动 `paper/**`**；未引用任何未逐字核到的条目。
  **单源核验的两条（Tent 的 ICLR 版本、Nado 的 DOI）已在文中显式标注。**

---

## 给 claude-0b 的三条建议

1. **方向 1 的 8 条足以回答复核者的 D2 风险**，尤其是 **[8] SHOT**（冻结特征提取器 + 只学头 + 无需源数据）
   与 **[1] Tent**（只动 BN 参数）—— **建议论文引这两条作答**。
2. **方向 2 请勿另造"last-layer recalibration"这个说法** —— 该方向在文献里以 **TTA / source-free DA** 命名。
3. **方向 3 我留了个明确的缺口**（Kumar 的方法学后续）。要补需换工具
   （Semantic Scholar 引文图 / ICLR 版独立记录）。**我没有把"没查到"写成"没有"。**
