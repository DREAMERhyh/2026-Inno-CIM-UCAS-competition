# 文献**内容**核查（不只核著录）—— Tent / SHOT

> 出处：claude-0b 派活（2026-09-24）。**起因是我在 `related_work_draft.md` 里把 SHOT 的机制写反了**
> （我写"冻结特征提取器、只学分类头"；**原文是"冻结分类器、学特征提取器"**）。
>
> **根因（claude-0b 指出，我认）**：
> 我用的是 **OpenAlex / Crossref**，**它们只核标题/作者/年份/卷期页码——核不到方法描述**。
> **"逐字核过"在我那里只覆盖了著录项。** 这是**方法上的盲区**，不是笔误。
>
> **本次纪律**：**内容只从原文取**（arXiv abs 页 / 若需要再抓 PDF 正文）；
> **元数据 API 只能证明"这篇存在"，不能证明"它做的是我们说的那件事"**；
> **一句话都不写进"已核实"，除非看到原文里支撑它的那句**。

---

## SHOT（arXiv 2002.08546，ICML 2020）

### 机制（**原文引自 arXiv abs 页，逐字**）

> "We propose a simple yet generic representation learning framework, named *Source HypOthesis
> Transfer* (SHOT). **SHOT freezes the classifier module (hypothesis) of the source model and
> learns the target-specific feature extraction module** by exploiting both information
> maximization and self-supervised pseudo-labeling to implicitly align representations from the
> target domains to the source hypothesis."

出处：`https://arxiv.org/abs/2002.08546`（本次 curl 抓取，2026-09-24）

### 我们/论文对它的描述

| 来源 | 原文 |
|---|---|
| **我的草稿**（`related_work_draft.md`） | 「**SHOT 冻结特征提取器、只学习分类头，且不需要访问源数据**」 |
| **论文**（`paper/journal/main.tex:181`） | 「SHOT 处理**不访问源数据**的域适应，它**冻结源模型的分类器、只学目标域的特征提取器**」 |

### 逐句比对

| 句 | 判定 |
|---|---|
| 不访问源数据 | **一致** ✓（原文："only a trained source model is available … without source data"） |
| **冻结的是哪个部分** | **我的草稿：不符（完全相反）** ❌ ；**论文：一致** ✓ |

**⇒ 我原稿写反了。原文是「冻结分类器（hypothesis）、学习特征提取器」；
我写成「冻结特征提取器、只学分类头」。**

**为什么这个错要紧**：它**恰好把 SHOT 与本文的关系说反了** ——
本文是「冻结特征、更新分类头」，而 SHOT 是「冻结分类器、更新特征」。
按我原稿的写法，本文与 SHOT 就成了**同一件事**，
那正是复核者 D2 担心的那个问题（"这不就是 target-domain 的 last-layer 重训吗"）的**反面**：
我等于**亲手把差异化抹掉了**。

**论文现在的写法反而是最锋利的**：

> 本文与这一支共享"只动极少量参数、也不需要源数据"的取向，
> **但动的对象正好相反**——本文冻结的是特征（主干物理上不可重训），更新的是分类头。

⇒ **这一句直接答掉 D2。**

---

## Tent（arXiv 2006.10726，ICLR 2021 spotlight）

### 机制（**原文引自 arXiv abs 页，逐字**）

> "In this setting of fully test-time adaptation the model has only the test data and its own
> parameters. We propose to adapt by test entropy minimization (tent): we optimize the model for
> confidence as measured by the entropy of its predictions. **Our method estimates normalization
> statistics and optimizes channel-wise affine transformations to update online on each batch.**
> … These results are achieved in one epoch of test-time optimization **without altering training**."

出处：`https://arxiv.org/abs/2006.10726`（本次 curl 抓取，2026-09-24）

### 我们/论文对它的描述

| 来源 | 原文 |
|---|---|
| **我的草稿** | 「**Tent 冻结全部主干，只更新批归一化的仿射参数**」 |
| **论文**（`main.tex:180`） | 「Tent 在测试时只最小化预测熵、**只更新批归一化的仿射参数，主干与分类器都不动**」 |

### 逐句比对

| 句 | 判定 |
|---|---|
| 最小化预测熵 | **一致** ✓（原文："we optimize the model for confidence as measured by the entropy of its predictions"） |
| 只更新（逐通道）仿射变换 | **一致** ✓（原文："optimizes channel-wise affine transformations"） |
| **⚠ 括号里少了一项** | 原文还说「**estimates normalization statistics**」——即 Tent **同时在线重估归一化统计量**。**论文与我都没有提这一项。** |
| "主干与分类器都不动" | **基本一致但不严谨** —— BN 的仿射参数**本身就在主干里**，说"主干不动"与"更新 BN 仿射参数"字面上相抵。**这是 TTA 文献里的通行简写**（"只更新 BN 参数"指不更新卷积/线性权重），**不构成实质错误** |

**⇒ Tent 的描述没有方向性错误，但有一处轻微省略（少说了"重估归一化统计量"）。**
**建议**（可选）：把「只更新批归一化的仿射参数」补成
「在线重估归一化统计量、只更新其仿射参数」。

---

## 回头审我自己的草稿

| 我原写 | 原文 | 处置 |
|---|---|---|
| 「SHOT **冻结特征提取器、只学习分类头**」 | 「**freezes the classifier … and learns the … feature extraction module**」 | ❌ **完全相反 ⇒ 已在 `related_work_draft.md` 改正**（并标注更正来源） |
| 「Tent **冻结全部主干**，只更新批归一化的仿射参数」 | 「estimates normalization statistics and optimizes channel-wise affine transformations」 | ⚠ 不严谨（BN 参数在主干里）⇒ 已改为与论文一致的写法 |
| 「该社群已确立『冻结主干、只更新极少数参数』是分布偏移下的有效范式」 | ✓ 两条摘要都支持 | 保留 |

> **⚠ 这条错误对本项目的意义**（记入方法学）：
> **元数据核实 ≠ 内容核实。** 我们为 `nat` 立的那条纪律（"每条必须逐字核"）
> **覆盖的是"这条文献存不存在"，不覆盖"它做的到底是不是我们说的那件事"。**
> **本次的 SHOT 反例说明：著录全对、机制写反，是可以同时发生的。**

---

## 四件套

- **里程碑**：Tent / SHOT 的**内容**核查 + 自查草稿
- **关键数字 + 出处**：
  - SHOT 机制原文（逐字）—— `https://arxiv.org/abs/2002.08546`
  - Tent 机制原文（逐字）—— `https://arxiv.org/abs/2006.10726`
  - **均来自 arXiv abs 页正文，未使用 OpenAlex/Crossref 的摘要字段**
- **判据命中哪个分支**：按三分法 —— **两条都"找到"**（原文可抓、逐字引到）。
  **SHOT：我的描述不符（已改）；论文的描述一致。Tent：我的与论文的都基本一致（一处轻微省略）。**
- **有没有触发红线**：**没有**。未动 `paper/**`；**未用元数据 API 代替原文**；
  **抓不到正文的地方如实说了**（无——两条摘要都抓到了）

---

## 一条建议（方法学，建议进项目纪律）

> **文献核实分两级，必须分开记**：
> **L1 著录核实**（标题/作者/年份/出处）—— 元数据 API 可做，成本极低；
> **L2 内容核实**（它到底做了什么）—— **只能读原文**，成本高，**且不能用 L1 顶替**。
> **引用一条文献去支撑某个具体断言时，至少要对该断言所依赖的那句话做 L2。**
>
> 本次 SHOT 的反例：**L1 全过、L2 相反。**
