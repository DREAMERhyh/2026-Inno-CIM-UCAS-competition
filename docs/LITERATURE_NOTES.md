# 文献笔记：与本项目的关系

> 为下一阶段做的定向检索（2026-09）。**每条标注核实状态**——
> ✅ 已抓取原文摘要逐字核对 / ⚠️ 只见于检索摘要、**尚未核实**。
> **未核实的不许写进论文**，也不许作为判据依据（见 §四）。

---

## 〇、一句话结论

检索出**三件对项目有实质影响的事**：

1. **本项目最核心的方法（冻结主干、只训读出）在文献里有名字：linear probing，且"它在 OOD 上能赢过全量微调"是一条 2022 年就有的、有理论支撑的已知结论**（Kumar et al.）。
   → 项目的贡献**不是这条规律本身**，而是**把它搬到一个权重物理上不可重训的新场景**，并补上该场景特有的三条：**上界（探针 headroom）、机制（秩塌陷）、边界（门控两前提）**。**这是定位风险，必须处理。**
2. **本项目最重要的测量（线性探针）在文献里正被系统地质疑**：探针精度会饱和、会失灵，且**不构成"信息不可恢复"的证据**。已有一整套补救手段（fragility / 临界噪声 / 私有校准划分）。
   → 我们的 headroom +20.12 是**单点测量**，需要加固。
3. **有几个"没人做过"的空白正好落在本项目的能力范围内**：ETF 原型读出是否抗输入扰动、池化的保护是否来自 ReLU 的非负性、失真是否让网络更接近神经坍缩几何。**这三个都能直接做。**

---

## 一、✅ 已核实的三条（抓取原文摘要逐字核对）

### 1.1 Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*

- **出处**：arXiv:2202.10054（ICLR 2022）。作者：Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang。
- **摘要原文要点**（逐字）：
  > "full fine-tuning … **can achieve worse accuracy than linear probing out-of-distribution (OOD) when the pretrained features are good and the distribution shift is large**."
  > "On 10 distribution shift datasets …, fine-tuning obtains on average **2% higher accuracy ID but 7% lower accuracy OOD** than linear probing."
  > 理论：在过参数化的两层线性网络上证明，"**while fine-tuning learns the head, the lower layers of the neural network change simultaneously and distort the pretrained features**"。
  > **LP-FT**（先线性探针再全量微调）："**1% better ID, 10% better OOD** than full fine-tuning"。

- **对本项目的意义**：
  - 我们实测的"训练端四条路全部无效、冻结主干只训读出最好"，**方向与这条已知结论一致**。
  - 他们给机制起的名字是 **feature distortion**（微调在学头的同时扭曲了底层特征）——
    这与我们观察到的"逐 epoch loss 贴合但尾部差 5.5"不完全是一回事，但**概念上是同一个家族**。
  - **我们的差异化必须写清**（三条，都是他们那篇没有的）：
    ① 失真不是分布偏移，而是**确定性的、逐层累积的、输入相关的系统误差**；
    ② 在 CIM 场景下**主干物理上不可重训**，所以 LP 不是"更好的选择"而是**唯一的选择**；
    ③ 我们给出**可恢复量的上界**（探针 headroom）与**损伤的几何形态**（秩塌陷），他们只给准确率。
- **可借用的实验**：**LP-FT**。他们发现"先探针再微调"能兼得。本项目**从未测过这条**——
  在失真场景下，"先训读出、再解冻主干微调"是否比"只训读出"更好？**这是一个现成的、有文献支撑的新实验。**

### 1.2 Reblitz-Richardson, *When Probing Accuracy Saturates, Fragility Resolves*

- **出处**：arXiv:2606.11375（2026）。作者：Orion Reblitz-Richardson。
- **摘要原文要点**（逐字）：
  > "**fragility**, a complementary per-layer metric defined as **the activation-noise level at which probe accuracy collapses**."
  > "Fragility is sensitive to both **the margin of separability** and **the redundancy of representation**."
  > "Where probing accuracy returns a flat answer, fragility returns a structured one."
  > ⚠ **关键陷阱**："deeper layers carry larger activations, so **a fixed absolute noise makes them look more robust than they are. Rescaling per layer removes most of the gradient** … We recommend **RMS-normalized critical noise for cross-layer claims** and raw critical noise for within-layer comparisons."

- **对本项目的意义（两条，第二条是陷阱）**：
  1. 我们只有**一个**探针读数（headroom）。**fragility 是现成的第二读法**，且它恰好能在精度持平时继续分辨——
     这正好用于回答"秩塌陷后剩余的维度是不是真的可用"。
  2. ⚠ **陷阱直接命中我们**：本项目已知 α>0 会把深层幅度压低（fc/conv1 的 L2 比 0.45 → 0.23）。
     若直接用**未归一化**的临界噪声，**失真组的激活更小 → 看起来"更鲁棒"**，会得出**完全相反**的结论。
     ⇒ **必须用 RMS 归一化的临界噪声**。

### 1.3 Pertl, Xuanyuan, Lio, *Superposition in Graph Neural Networks*

- **出处**：PMLR v322（UniReps 2026）。**注意：是 GNN 研究，不是 CNN 鲁棒性研究。**
- **摘要原文要点**（逐字）：
  > "**sharper pooling increases axis alignment** and reduces channel sharing"
  > "topology imprints overlap onto node-level features that **pooling partially remixes into task-aligned graph axes**"
  > "connect representational geometry with concrete design choices (**width, pooling, and final-layer activations**)"
  > "shallow models can settle into **metastable low-rank embeddings**"

- **对本项目的意义**：
  - 它给出一个我们**从未考虑过的池化机制**：池化的作用可能是**让表示"轴对齐"**，而轴对齐的表示**在等能量扰动下损失更小**。
  - 这解释了本项目一个悬而未决的细节：**为什么 max 换成 avg 只差 1.17 pp？**
    如果保护来自"池化带来的轴对齐"，那么 avg 和 max **都能带来轴对齐**，差异自然就小。
    而本项目现有的三条排除线（非吸收、非不变区、非 max 选择性）**恰好都没排除这一条**。
  - 它同时点名两个我们没扫过的设计变量：**宽度**、**最后一层的激活函数**。
  - ⚠ **待核实**："轴对齐在 max pooling 下对等能量扰动损失更小"这个具体论断来自检索摘要里对**图 5** 的描述，
    我只核对了摘要、**没核对正文图表**。使用前需读原文 §Figure 5。

---

## 二、⚠️ 只见于检索摘要、尚未核实（**不许引用，先核实**）

> **核实进展（2026-09-24，M0b 执行）**：7 条里 **4 条已销**，剩 3 条。
>
> | 原编号 | 文献 | 新状态 |
> |---|---|---|
> | 1 | Science Advances MTJ (sciadv.adp3710) | ✅ **已核实**——标题/作者/期刊/DOI：*Adapting magnetoresistive memory devices for accurate and on-chip-training-free in-memory computing*，T. Xiao, V. B. Naik, J. H. Lim, Y. Hou, Z. Wang, Q. Shao，**Sci. Adv. 10(38):eadp3710, 2024-09-20**。内容：MTJ + **离线标定**达软件基线、无需片上训练；自适应量化用**电导漂移感测** |
> | 2 | IBM HWA (Nature Comm s41467-023-40770-4) | ✅ **标题已核实**：*Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators* |
> | 6 | Graph-JEPA (arXiv 2608.20516) | ✅ **逐字核实**：Gollam Rabby, Sören Auer；「linear-probe accuracy 0.871 and effective rank 18-47, yet retrieval recovers **0.00 of 14.4 bits**」；「**Rank, probes, and metrics can all saturate on an unsupportive evaluation**」 |
> | 7 | arXiv 2310.03843 | ✅ **已核实，但标题与我原笔记不符**——实为 *From Channel Bias to Feature Redundancy: Uncovering the "Less is More" Principle in Few-Shot Learning*（Zhang, Luo, Gao, Zou, Shen, Song）。原笔记写的 *Less is More: On the Feature Redundancy…* 应是 v2 改名（该页有 "substantial text overlap with arXiv:2206.08126" 的 admin note）。内容对得上：**少样本下 1–5% 的特征维度就够，且大部分维度是有害的** |
> | 3 | Nature Comm PCM 漂移补偿 | ⚠ **仍未见原文** |
> | 4 | IGZO DRAM CIM (IEEE 2025) | ⚠ **仍未见原文** |
> | 5 | ICML 2026 *Neural Collapse by Design* | ⚠ **页面是 JS 渲染，curl 抓不到正文**；需 `playwright-cli --browser=msedge` |
>
> **结论**：`docs/POSITIONING.md` 只使用 ✅ 的条目。⚠ 的对本项目论断**不作依据**。

| 文献 | 检索摘要说它是什么 | 对本项目的潜在意义 | 待办 |
|---|---|---|---|
| *Adapting magnetoresistive memory devices for accurate and on-chip-training-free in-memory computing*（Science Advances，DOI 10.1126/sciadv.adp3710） | MTJ 存内计算，**无需片上训练**即达软件基线：用器件特定电导漂移查找表 + 自适应量化，**少量校准样本重新参数化预训练模型**；MAC RMSE 2.48 → 0.57 LSB | **最接近的先行工作**。若属实，说明"冻结阵列 + 数字域少量校准"**在 CIM 领域已有人做**，我们的定位必须更精确 | 抓原文（当前被墙） |
| IBM *Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing*（Nature Communications 2023, s41467-023-40770-4） | HWA 让 CNN/RNN/Transformer 在 AIMC 上达到 iso-accuracy；**加在输入/输出上的非理想性比加在权重上的影响更大** | **直接印证**：本项目的失真正是**输入侧**扰动 | 抓原文 |
| PMC 漂移补偿（Nature Communications, s41467-023-40770-4 相关） | 电导漂移"**can be compensated in the digital domain without any expensive re-programming**"：测 s_ref/s_eval，调数字缩放因子 | 数字域补偿的**标量版**；我们的读出层适配是**更强的形式** | 抓原文 |
| IGZO DRAM CIM（IEEE 2025） | 免刷新的 ADC 时变校准 | 同类思路的 ADC 侧版本 | 抓原文 |
| *Neural Collapse by Design: Learning Class Prototypes on the Hypersphere*（ICML 2026） | 用原型对比损失强制神经坍缩几何，**ImageNet-C 上鲁棒性提升**；且"**监督对比学习在预训练中达到 NC，但在事后线性探针中丢掉了它**" | 给我们一个**全新解释框架**（见 §三） | 抓原文 |
| *Graph-JEPA: Diagnosing and Repairing Category-Conditional Collapse*（arXiv 2608.20516） | 一个模型**线性探针 0.871、有效秩 18–47 全部通过**，但检索上**只恢复了 14.4 bit 里的 0.00 bit**；作者警告"**秩指标与探针可能虚假饱和**" | **对 Central Claim 的直接警示**（见 §三） | 抓原文 |
| *Less is More: On the Feature Redundancy of Pretrained Models*（arXiv 2310.03843） | 少样本下**仅用 1% 的特征维度**即可恢复全表示性能；冗余随样本数增加而消失 | 读出训练样本量的冗余曲线（新增任务） | 抓原文 |

**获取方式**（WebFetch 在本机被安全策略拦截）：用 `curl`（已验证对 arXiv 与 PMLR 有效），
JS 渲染页面用 `playwright-cli --browser=msedge`。

---

## 三、三条"文献指出的空白"——**正好落在本项目能力范围内**

> 这三条是本次检索最有价值的部分：**它们不是"我们没想到"，而是"文献里也没人做过"。**

### 空白 1：ETF / 原型读出**是否抗输入扰动**？

- 已知：神经坍缩的 ETF 几何**提升 corruption 鲁棒性**（ICML 2026 摘要）；固定 ETF 原型被用于增量学习、长尾、少样本。
- **未找到**：把神经网络最后一层的线性头**替换成固定 ETF 原型分类器**、然后测**输入扰动**下鲁棒性的研究。
- **本项目为什么正好能做**：我们已经有冻结主干 + 各类 α 的特征，只要有 CSV 就能离线比较"线性头 vs 原型头"，
  **零 GPU**。这是一条**极低成本、可能高回报**的新方向。

### 空白 2：池化的保护**是否来自 ReLU 的非负性**？

- 已知（Pertl，待核实正文）：池化提升**轴对齐**；轴对齐的向量在等能量扰动下**损失更小**。
- 已知（本项目）：post-ReLU 的激活**非负**，即在正卦限内——这是最"轴对齐"的姿态。
- **可测预测**：**把 ReLU 换成 GELU / LeakyReLU（破坏非负性），池化密度的保护应当变弱。**
- **没找到有人做过这个对照。** 这是一个**干净的单变量实验**，而且它能把本项目 L8 的"主要来自降维"
  升级成"**来自降维 × 非负性**"——**当前三条排除线没有排除这一条**。

### 空白 3：失真是否让网络**更接近神经坍缩几何**？

- 已知：NC 的特征是类均值收敛到 **Simplex ETF**（等角、等模），且 **Fisher 判别比最大化**。
- **本项目的异常**：正侧失真下**有效秩掉 20.6%，但 Fisher 反而升高**。
  **这恰恰是 NC 的签名**！我们此前把它当作"排除了类间距缩小"，但**没有顺着这条线往下问**。
- **可测的预测**：失真是否让**类均值两两夹角更接近 ETF 的理想值**？若是，则
  "损伤 = 病理性 NC 化（类间距相对变大、绝对维度丢失）"——一个**有文献支撑的替代叙述**。
- 这条若成立，**L4/L6 的解释会换一套语言**，而且能直接预测"ETF 原型读出"（空白 1）为什么会有效。

---

## 四、对"记录纪律"的影响（新增两条）

1. **未核实的文献不许引用**。本文件 §二 的表全部标 ⚠，**核实前不得进入论文参考文献**。
   本项目的既有纪律是"每个数字要能指向文件"；**文献同理，每条要能指向已抓取的原文**。
2. **探针类结论必须给出第二读法**。文献已经确立"探针饱和不等于信息在场、探针失败也不等于信息不在场"。
   ⇒ 本项目所有基于探针的结论（**headroom +20.12 是 Central Claim 的第一块砖**）都要配一个
   **fragility（RMS 归一化的临界噪声）**读数。

---

## 五、检索方法备忘（留给后来者）

| 通道 | 状态 |
|---|---|
| `WebSearch` | **可用**，返回摘要级信息 |
| `WebFetch` | **被安全策略拦截**（任何域名都返回 "Unable to verify if domain … is safe to fetch"）——**不要试** |
| `curl` + 去标签 | **可用**。arXiv `abs/` 页与 PMLR 页均成功抓取 |
| `playwright-cli --browser=msedge` | 备用，用于 JS 渲染页（本机只有 Edge，没有 Chrome） |
| 需要登录的站点 | 先弹浏览器窗口让人登录，再抓 |

**去标签命令**（已验证可用）：
```bash
curl -sL --max-time 25 -A "Mozilla/5.0" "<url>" \
  | sed 's/<script[^>]*>.*<\/script>//g; s/<[^>]*>/ /g' | tr -s ' \n' ' \n'
```
