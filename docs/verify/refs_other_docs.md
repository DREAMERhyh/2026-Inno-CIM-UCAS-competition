# 其余三份文档的参考文献逐条核查

> ⏱ **快照声明**：本文件的**所有计数类结论**（"共 n 条""`\cite` 计数 1/4/0""n 条未引用"）
> **只在该时点成立**——最后写入 **2026-09-24 16:32**（三份文档当时都有人在改）。
> 文献**著录本身**的结论（✅/❌）**不受时间影响**（DOI/页码是外部事实）；
> 但 **`\cite` 计数会随写者补引用而失效**。凡位置引用一律附**原文字符串**，行号会漂。

> **派活**：claude-0b（2026-09-24 晚）。**目的**：把 `nat` 那条的账查清之后，回答"**其余三份里还有没有第二个 `nat`**"。
> **对象**：`paper/technical-report/main.tex`、`paper/competition/main.tex`、`ucas-cod-lab-report/main.tex`。
> **方法**：**Crossref / OpenAlex / arXiv API 结构化比对**（作者 / 标题 / 出处 / 年份 / 卷期页码 / DOI），**不依赖搜索结果摘要**；
> 难索引的（ICLR、技报）走 arXiv API 与官方 URL。
> **纪律**：**只读**——三份文档一个字都没写（`technical-report`/`competition` 正被 claude-89 改着）。
> 引用位置一律**附原文字符串**，行号只作辅助（这些文件今天被改过，行号会漂）。
> **产物**：本文件（新建）+ `reviewer_round2.md` 追加一节。

---

## 0. 先说三份的共同结构（这决定了核查的工作量）

三份文档**用的是同一组 8 条**（就是 `paper/health-check-report.md` §1.1 说的"仓库里仅有的 8 篇已核实文献"）：

| 文档 | bibitem 数 | 与其余两份的差异 |
|---|---|---|
| `technical-report/main.tex` | **8** | 无（顺序不同） |
| `competition/main.tex` | **8** | 无（顺序不同） |
| `ucas-cod-lab-report/main.tex` | **8** | 无（**期刊名写全称**、用 `-` 而非 `--`） |

⇒ 8 条里 **7 条我在上一轮（journal）已经逐条联网核实过**，本轮的增量工作是：
① 用**结构化 API**再核一遍（上轮用的是 WebSearch 摘要，本轮要求升级）；
② **独立复现 `nat` 的"结构性不存在"**；
③ 补上三份都缺的 `lpft`/`mtjcal`/`gjepa` 三条的**缺失影响**评估。

**核查用的 API 与可达性（如实记）**：

| 通道 | 状态 | 用途 |
|---|---|---|
| Crossref REST（`api.crossref.org`） | ✅ 通 | 主力：按标题检索引擎 + **整期页码地图** |
| OpenAlex（`api.openalex.org`） | ✅ 通 | 兜底（arXiv 预印本、无 DOI 条目） |
| arXiv API（`export.arxiv.org`） | ✅ 通 | ICLR 类无 DOI 条目（VGG） |
| 官方 URL（`curl -I`） | ✅ 通 | 无 DOI 技报（CIFAR） |
| IEEE Xplore / WebFetch(arxiv.org) | ❌ 本机不可用 | **未走通，不计入"落空"**；Crossref 元数据来自出版商本身，可替代 |

---

## （一）`paper/technical-report/main.tex`：共 8 条

> 引用位置：`\begin{thebibliography}` 段（我读到的行号 1157–1164），逐条附原文。

**[1]** `Yu S, Chen P Y. Emerging memory technologies: recent trends and prospects[J]. IEEE Solid-State Circuits Magazine, 2016, 8(2): 43--56.`
→ **✅ 逐字核到**
Crossref：`Emerging Memory Technologies: Recent Trends and Prospects`，作者 **Yu Shimeng（2 位）**，`IEEE Solid-State Circuits Magazine`，**2016，8(2): 43-56**，[10.1109/MSSC.2016.2546199](https://doi.org/10.1109/MSSC.2016.2546199)。作者、刊名、年、卷、期、页**逐字一致**。

**[2]** `Ankit A, El Hajj I, Chalamalasetti S R, et al. PUMA: a programmable ultra-efficient memristor-based accelerator for machine learning inference[C]//ASPLOS. Providence: ACM, 2019: 715--731.`
→ **✅ 逐字核到**
Crossref：`PUMA`，作者 **Ankit Aayush（11 位）**，`Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems`（=ASPLOS 2019，Providence），**2019，715-731**，[10.1145/3297858.3304049](https://doi.org/10.1145/3297858.3304049)。**一致**（"et al." 覆盖 11 位作者，合规）。

**[3]** `Zhu M, Liu Y, Wang Z, et al. Nonlinearity-aware training for accurate and robust compute-in-memory neural networks[J]. IEEE TCAD, 2022, 41(11): 3961--3973.`
→ **❌ 结构性不存在**（详见 §四 专段，含独立复现的页码地图）

**[4]** `Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]//CVPR. Salt Lake City: IEEE, 2018: 2704--2713.`
→ **✅ 逐字核到**
Crossref：`Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference`，作者 **Jacob Benoit（8 位）**，`2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition`，**2018，2704-2713**，[10.1109/CVPR.2018.00286](https://doi.org/10.1109/CVPR.2018.00286)。**一致**。

**[5]** `Wannamaker R A, Lipshitz S P, Vanderkooy J, et al. A theory of nonsubtractive dither[J]. IEEE Transactions on Signal Processing, 2000, 48(2): 499--516.`
→ **✅ 逐字核到**
Crossref：`A theory of nonsubtractive dither`，作者 **Wannamaker R.A.（4 位）**，`IEEE Transactions on Signal Processing`，**2000，48(2): 499-516**，[10.1109/78.823976](https://doi.org/10.1109/78.823976)。**一致**（"et al." 覆盖第 4 位 Wright J N）。

**[6]** `He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//CVPR. Las Vegas: IEEE, 2016: 770--778.`
→ **✅ 逐字核到**
Crossref：`Deep Residual Learning for Image Recognition`，作者 **He Kaiming（4 位）**，`2016 IEEE Conference on Computer Vision and Pattern Recognition`，**2016，770-778**，[10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)。**一致**（会场名写"CVPR"缩写，可接受）。

**[7]** `Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[C]//ICLR. San Diego: ICLR, 2015: 1--14.`
→ **✅ 真实**（Crossref **不索引 ICLR 2015**，改用 arXiv + OpenAlex 核）
- arXiv [1409.1556](https://arxiv.org/abs/1409.1556)：`Very Deep Convolutional Networks for Large-Scale Image Recognition`，作者 **Karen Simonyan, Andrew Zisserman**，2014-09-04 提交 ✓
- OpenAlex 亦命中同一标题（2014 arXiv 版 / 2015 ICLR 版）
- ⚠ **小注（不是错误）**：ICLR **没有正式页码**，"1--14" 不是官方值（部分文献表沿用，非编造）。若想更规范，可改引 arXiv 号或写"ICLR 2015 Conference Track Proceedings"。

**[8]** `Krizhevsky A. Learning multiple layers of features from tiny images[R]. Toronto: University of Toronto, 2009.`
→ **✅ 真实**（技报**无 DOI**，Crossref 不索引）
- 官方 PDF 现址 `curl -I` → **HTTP 200, application/pdf**：<https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>（重定向到 `cave.cs.toronto.edu/kriz/`）
- 著录（作者/标题/机构/年）与全领域通行 BibTeX 一致 ✓

**技术报告小结**：8 条 → **✅ 7 / ❌ 1**。

---

## （二）`paper/competition/main.tex`：共 8 条

> 引用位置：`\begin{thebibliography}` 段（我读到的行号 701–708）。

**逐条结果与（一）完全相同**（同一组 8 条，字符串一致）：

| # | 键 | 结果 |
|---|---|---|
| 1 | `cim_survey` | ✅ 10.1109/MSSC.2016.2546199 |
| 2 | `puma` | ✅ 10.1145/3297858.3304049 |
| 3 | `vgg` | ✅ arXiv:1409.1556（页码 1-14 非官方，见上） |
| 4 | `resnet` | ✅ 10.1109/CVPR.2016.90 |
| 5 | `cifar` | ✅ 官方技报 PDF HTTP 200 |
| 6 | **`nat`** | **❌ 结构性不存在**（同上，见 §四） |
| 7 | `quant` | ✅ 10.1109/CVPR.2018.00286 |
| 8 | `dither` | ✅ 10.1109/78.823976 |

**竞赛版小结**：8 条 → **✅ 7 / ❌ 1**。

> **对竞赛版特别有用的一点**：`nat` 在这份里**从未被正文引用**（见 §五），
> 所以**删掉它不需要动任何一句话**——这是三份里最容易处理的。

---

## （三）`ucas-cod-lab-report/main.tex`：共 8 条

> 引用位置：`\section{参考文献}` 下的 `\begin{thebibliography}{99}`（我读到的行号 1088–1095）。

**逐条结果**：与（一）（二）**同一组 8 条、同样的核查结论**——

- `resnet` ✅ / `vgg` ✅ / `puma` ✅ / `quant` ✅ / `cifar` ✅ / `cim_survey` ✅ / `dither` ✅
- **`nat` ❌ 结构性不存在**（这一版把刊名写成全称 `IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems`，**内容相同**，因此同样是 ❌）

**唯一的形式差异（不是错误）**：本文件用**全称刊名**与 `-`（单连字符），其余两份用缩写 + `--`。著录条目本身**没有对错之分**。

**实验报告版小结**：8 条 → **✅ 7 / ❌ 1**；**且 8 条全部未被正文引用**（见 §五）。

---

## （四）`nat` 专段：**独立复现**"结构性不存在"

> 上一轮我给的结论是"5 条检索路径落空、无法核实"（当时标 ⛔/未核实）。
> 本轮换**出版商元数据**这条路，拿到了**结构性证据**。

### 方法：拉整期页码地图（Crossref）

Crossref 的 IEEE TCAD 期刊 ISSN `0278-0070`，取**印刷日期 2022-11 的全部条目**：

```
GET https://api.crossref.org/journals/0278-0070/works
    ?rows=200&filter=from-print-pub-date:2022-11-01,until-print-pub-date:2022-11-30
```

→ 返回 **133 条**，客户端筛出 `volume=41 & issue=11` 全部命中，页码区间 **3567–5135**（+ 封面 C1–C3）。
**目标页附近的连续页码地图**（按起始页排序，**无任何空隙**）：

| 页码 | DOI | 标题（截断） |
|---|---|---|
| 3934–3945 | 10.1109/tcad.2022.3197540 | Architecting Decentralization and Customizability in DNN Accel… |
| 3946–3956 | 10.1109/tcad.2022.3197542 | LLSM: A Lifetime-Aware Wear-Leveling for LSM-Tree on NAND Flash |
| **3957–3968** | **10.1109/tcad.2022.3199146** | **Characterizing the Effect of Deadline Misses on Time-Triggered…** |
| **3969–3980** | **10.1109/tcad.2022.3197971** | **Efficient Backward Reachability Using the Minkowski Difference…** |
| **3981–3992** | **10.1109/tcad.2022.3197978** | **Throughput Maximization in Wireless Communication Systems…** |
| 3993–4003 | 10.1109/tcad.2022.3197980 | Amphis: Managing Reconfigurable Processor Architectures… |

### 结论

论文里那条 `41(11): 3961--3973` **落在这三篇中间被完全占满的区间里**：
**3961–3968 属于"Deadline Misses"那篇，3969–3973 属于"Backward Reachability"那篇**——
**没有任何 13 页的空间**留给第三条论文。⇒ **❌ 结构性不存在。**

**交叉印证（不同路径、同一结论）**：
- 另一会话拉 **IEEE Xplore 该期目录**，得到同样的 `3957–3968 / 3969–3980 / 3981–3992` 三篇连续铺满；
- 我这次走 **Crossref 出版商元数据**，独立得到同一张地图；
- **按标题精确检索**（Crossref `query.bibliographic`）**反复命中的是另一篇真实论文**：
  **NEAT: Nonlinearity Aware Training…**，Bhattacharjee 等 4 位，
  **IEEE TCAD 2022, 41(8): 2625–2637**，[10.1109/TCAD.2021.3109857](https://doi.org/10.1109/TCAD.2021.3109857)——
  **同主题、同年、同刊，但作者/卷/期/页全不同**。

**残留不确定性（如实写）**：Crossref 反映的是**出版商已存缴**的元数据。理论上不能 100% 排除
"IEEE 漏存缴某篇"的极端情形；但（a）IEEE 对 TCAD 是全量存缴，（b）三条独立路径给出同一结论，
（c）标题检索指向一篇可解释的"近名"真实论文 ⇒ **判断为 ❌，且这个判断的分量已足够支撑处理决定**。

---

## （五）⚠ 一条**本轮新发现**：这三份的参考文献**几乎没被正文引用**

> 这不是"文献真假"问题，是"参考文献表与正文对不上"问题。但它直接影响 `nat` 该怎么处理，
> 而且**与 `paper/health-check-report.md` §4 的一条结论不符**，所以单列。

**实测（`\cite` 出现的次数，逐份点的）**：

| 文档 | 文中 `\cite` 命令数 | 被引用的键 | **从未被引用的 bibitem** |
|---|---|---|---|
| `technical-report` | **1**（`~\cite{cifar}`，我读到第 162 行，紧接"数据集：CIFAR-10（10 类）与 CIFAR-100（100 类）"） | `cifar` | **7 条**：`cim_survey` `puma` `nat` `quant` `dither` `resnet` `vgg` |
| `competition` | **4**（第 79 行 `\cite{cim_survey,puma}`；131 `cifar`；136 `vgg`；137 `resnet`） | `cim_survey` `puma` `cifar` `vgg` `resnet` | **3 条**：**`nat`** `quant` `dither` |
| `ucas-cod-lab-report` | **0**（全文不含 "cite" 这个字符串） | —— | **8 条（全部）** |
| （对照）`journal` | 多处 | 11 条**全部**被引用 | 0 |

**为什么这条值得记**：
1. **`paper/health-check-report.md` §4 写的是**「参考文献键完整性（`\cite` 集合 vs `\bibitem` 集合求差）→
   三份稿件：**缺失 0、未引用 0**」——**对期刊版成立，对另两份不成立**（技术报告 7 条、竞赛版 3 条未引用）。
   最可能的解释是那次检查只做了**一个方向**（"引用的键是否都在 bibitem 里"= 缺失 0 ✓），
   "未引用"那半句没有真的算。**建议把这条结论改掉**（与第一轮我发现的"152 行全部通过、其中 94 行 n/a"是同一类：
   **检查的口径被说满了**）。也可能是文件在这之后被大改过（技术报告从 18 页长到 22 页），无论哪种，**当前版本的事实是上表**。
2. **它决定 `nat` 的修法成本**：竞赛版里 `nat` **从未被引用** ⇒ **直接删掉这一条即可，正文一字不用改**；
   技术报告与实验报告版同样从未引用 `nat` ⇒ 同样是"删一条"。
3. **实验报告版的 8 条全部未引用** ⇒ 那张参考文献表目前是**装饰性的**（读者无法知道哪句话对应哪条）。
   对内部报告可以接受，但**既然要在报告里放参考文献表，建议至少把 VGG/ResNet/PUMA/CIFAR 四处补上 `\cite`**。

---

## （六）三份都没有 `lpft` / `mtjcal` / `gjepa` —— 但技术报告**在做它们支撑的论断**

期刊版有 11 条（比这三份多 `lpft`、`mtjcal`、`gjepa`）。三份都只有 8 条。
**光看条数不是问题**；问题是**技术报告里有段落正在做这三条文献支撑的论断而没有引用**。

我在 `paper/technical-report/main.tex` 读到的原文（**附原文字符串**，行号仅辅助）：

> （第 105–106 行附近）「本项目的核心方法——冻结主干、只重训最后的线性层——**不是一个新方法**，
> 它是迁移学习里的 linear probing；而「预训练特征好、偏移大时线性探针可胜过全量微调」
> 是一条**有理论与十数据集实证支撑的已知结论**。」
> （第 114 行附近）「本报告同时**主动记录**了本项目最依赖的测量手段（线性探针）正被**文献警示可能虚假饱和**，
> 以及对应的加固方案。」

- 「有理论与十数据集实证支撑」= 期刊版引的 **LP-FT（Kumar et al., ICLR 2022, arXiv:2202.10054）**；
- 「被文献警示可能虚假饱和」= 期刊版引的 **Graph-JEPA（arXiv:2608.20516）**。

⇒ **技术报告说了"有文献"却没给出处**。这与我第一轮在期刊版抓到的 **A14（fragility 方法未引出处）** 是同一类问题，
只不过这里是**跨文档**的：**同一句话在期刊版有引用、在技术报告版没有**。
**建议**：把 `lpft` 与 `gjepa` 两条也加进技术报告的参考文献表并在该段引用（`mtjcal` 对应"CIM 领域「冻结阵列＋数字域补偿」已有先例"那句，同样建议补）。

---

## （七）汇总

```
technical-report：共 8 条 → ✅ 7 / ⚠️ 0 / ❌ 1 (nat) / ⛔ 0
competition     ：共 8 条 → ✅ 7 / ⚠️ 0 / ❌ 1 (nat) / ⛔ 0
ucas-cod-lab-report：共 8 条 → ✅ 7 / ⚠️ 0 / ❌ 1 (nat) / ⛔ 0
─────────────────────────────────────────────
合计 24 条（去重后 8 条）： ✅ 21 / ⚠️ 0 / ❌ 3 (同一把 nat 出现在三份) / ⛔ 0
```

**需要作者决定的（❌ 与 ⚠️ 逐条列出）**：

| 级别 | 条目 | 出现在 | 事实 | 处理建议 |
|---|---|---|---|---|
| ❌ | `nat` = Zhu M, Liu Y, Wang Z, et al. *Nonlinearity-aware training…*, IEEE TCAD **41(11): 3961–3973**, 2022 | **四份都有**（journal + 这三份） | **该页码被两篇别的论文占满**（3957–3968 / 3969–3980），标题目录里无此条；近名真实论文是 **NEAT**（TCAD 41(8): 2625–2637） | 三份里**都从未被正文引用** ⇒ **直接删这条即可，正文一字不动**；若要保留一个 NAT 出处，换成 NEAT（10.1109/TCAD.2021.3109857，已核实） |
| ⚠️ | **0 条** | —— | 其余 7 条**作者/标题/出处/年/卷/期/页逐字一致** | 无需处理 |

**唯一的 ⚠️ 级候选（我判为不构成 ⚠️，如实登记供裁决）**：`vgg` 的页码 `1--14`。
标题/作者/会场/年**全对**，只是 **ICLR 无正式页码**，该范围非官方值（有文献表沿用先例）。
⇒ 按派活的定义（"编号对但标题/著录对不上"）**不算 ⚠️**，但若你们的标准更严，它就是 1 条。

**⛔（检索能力不足）**：**0 条**。本轮所有条目都走到了可引用的证据源（Crossref/OpenAlex/arXiv/官方 URL）。
需要说明的是：**IEEE Xplore 直连与 WebFetch(arxiv.org) 在本机不可用**，我改用 Crossref 出版商元数据替代——
**这条路走通了，所以不记为"落空"**（照 claude-0b 的规矩：走不通的通道单列，不计入条目结论）。

---

## 四件套

**里程碑**：
- 三份文档的**全部 8 条**（去重后）用 **Crossref 主检 + OpenAlex/arXiv/官方 URL 兜底**逐条结构化核对完毕；
- **独立复现了 `nat` 的"结构性不存在"**：用 Crossref 拉出 TCAD 41(11) 整期 133 条的页码地图，
  证明 `3961–3973` 落在 **3957–3968 / 3969–3980** 两篇真实论文的页内，**没有第三个位置**；
  与另一会话走 Xplore 目录得到的结论**完全一致**；
- **新发现**：这三份的参考文献**几乎没被引用**（技术报告 7/8、竞赛版 3/8、实验报告版 8/8 未引用），
  且这与 `health-check-report.md` §4 的"未引用 0"**不符**；
- **新发现**：技术报告做了 LP-FT 与探针饱和的论断却无引用（同一句话在期刊版有引用）。

**关键数字 + 出处文件**：
| 数字/事实 | 出处 |
|---|---|
| 三份各 8 条、条目字符串同源 | `paper/technical-report/main.tex:1157–1164`、`paper/competition/main.tex:701–708`、`ucas-cod-lab-report/main.tex:1088–1095`（行号会漂，按字符串检索） |
| TCAD 41(11) 共 133 条、页码 3567–5135 | Crossref `journals/0278-0070/works` + `filter=from-print-pub-date:2022-11-01,until-print-pub-date:2022-11-30` |
| **3957–3968** 占位论文 | 10.1109/tcad.2022.3199146 |
| **3969–3980** 占位论文 | 10.1109/tcad.2022.3197971 |
| **3981–3992** 占位论文 | 10.1109/tcad.2022.3197978 |
| 近名真实论文 NEAT = **41(8): 2625–2637** | [10.1109/TCAD.2021.3109857](https://doi.org/10.1109/TCAD.2021.3109857) |
| 7 条已核实的 DOI | [10.1109/MSSC.2016.2546199](https://doi.org/10.1109/MSSC.2016.2546199) / [10.1145/3297858.3304049](https://doi.org/10.1145/3297858.3304049) / [10.1109/CVPR.2018.00286](https://doi.org/10.1109/CVPR.2018.00286) / [10.1109/78.823976](https://doi.org/10.1109/78.823976) / [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90) + arXiv:[1409.1556](https://arxiv.org/abs/1409.1556) + CIFAR 技报 PDF（HTTP 200） |
| `\cite` 计数（1 / 4 / 0） | `grep -o "cite" <file> \| wc -l`；原始行：技术报告第 162 行 `~\cite{cifar}`；竞赛版第 79/131/136/137 行；实验报告版 0 处 |

**判据命中哪个分支**：
- 派活要的是回答"**其余三份里还有没有第二个 `nat`**" ⇒ **答案是"没有第二个；只有同一把 `nat` 出现在三份里"**。
  去重后 8 条中 **❌ 恰好 1 条（就是 `nat`）**，且它在四份文档里**同源同字符串**。
- 四级分类的落点：**✅ 21 / ⚠️ 0 / ❌ 3 / ⛔ 0**。

**有没有触发红线**：
- **没有**。三份文档（含正被 claude-89 改着的两份）与 `ucas-cod-lab-report/**` **全程只读**，
  未写一个字；未改任何数据文件；**未用 GPU**；未 commit；产物只有本文件与 `reviewer_round2.md` 的追加节。
- **存量内容命中一条红线**：仍是 **红线 9「引用前必须验证文献」**——
  `nat` 在三份里都以完整著录形态存在，而它的页码位置被别的论文占着。
  本轮把"要不要处理"的证据补齐了：**三份里它都从未被正文引用，所以"删掉这一条"是零风险的修法**。
